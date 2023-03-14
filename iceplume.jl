using Oceananigans
using Oceananigans.Units
using Printf
using CUDA: has_cuda_gpu, @allowscalar
using Statistics: mean


#++++ High level options
mass_flux = true
LES = true
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
#----


#++++ Construct grid
const Lx = 800
const Ly = 800
const Lz = 800
Nx = 16
Ny = Int(Nx/2)
Nz = Nx

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (-Ly/2, +Ly/2),
                       z = (0, Lz),
                       topology = (Bounded, Periodic, Bounded))
@info "Grid" grid
#----


#++++ Creates a dictionary of simulation parameters
# (Not necessary, but makes organizing simulations easier and facilitates running on GPUs
params = (N²₀ = 1e-6, # 1/s (stratification frequency)
          z₀ = 300, # m
          σz_west = 10, # m
          ℓ₀ = 0.1, # m (roughness length)
          σ = 1/30minutes, # s (relaxation rate for sponge layer)
          uₑᵥₐᵣ = 0.001 # m/s (velocity variation along the z direction of the east boundary)
          )

params = merge(params, (; b₁_west = params.N²₀ * Lz))
#----

#++++ Interior conditions
b∞(z, p) = p.N²₀ * z # Linear stratification in the interior (far from ice face)
#----



#++++ Western BCs
b_west(y, z, t, p) = b∞(z, p) + p.b₁_west / (1 + exp((z-p.z₀)/p.σz_west))

if mass_flux
    params = (; params..., u₁_west = 1e-3) # m/s
else
    params = (; params..., u₁_west = 0) # No mass flux
end
u_west(y, z, t, p) = p.u₁_west / (1 + exp((z-p.z₀)/p.σz_west))

#++++ Drag BC for v and w
const κ = 0.4 # von Karman constant
x₁ₘₒ = @allowscalar xnodes(Center, grid)[1] # Closest grid center to the bottom
cᴰ = (κ / log(x₁ₘₒ/params.ℓ₀))^2 # Drag coefficient

@inline drag_w(x, y, t, v, w, p) = - p.cᴰ * √(w^2 + v^2) * w
@inline drag_v(x, y, t, v, w, p) = - p.cᴰ * √(w^2 + v^2) * v

drag_bc_w = FluxBoundaryCondition(drag_w, field_dependencies=(:v, :w), parameters=(cᴰ=cᴰ,))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:v, :w), parameters=(cᴰ=cᴰ,))
#----

#----



#++++ Eastern BCs
if mass_flux # What comes in has to go out
    params = (; params..., u_out = mean(u_west.(0, grid.zᵃᵃᶜ[1:Nz], 0, (params,))))
else
    params = (; params..., u_out = 0)
end

# The function below allows a net mass flux out of exactly u_out, but with variations in the form of
# a sine function. After a mean in z, this function return exactly u_out.
u_east(y, z, t, p) = p.u_out + p.uₑᵥₐᵣ*sin(-2π*z/grid.Lz) 
#----


#++++ Eastern sponge layer 
# (smoothes out the mass flux and gets rid of some of the build up of buoyancy)
@inline function west_mask_cos(x, y, z)
    x₀ = 600
    x₁ = 700 
    x₂ = 800

    if x₀ <= x <= x₁
        return 1/2 * (1 - cos( π*(x-x₀)/(x₁-x₀) ))
    elseif x₁ < x
        return 1.0
    else
        return 0.0
    end
end

@inline sponge_u(x, y, z, t, u, p) = -west_mask_cos(x, y, z) * p.σ * (u - u_east(y, z, t, p)) # Nudges u to u_east
@inline sponge_v(x, y, z, t, v, p) = -west_mask_cos(x, y, z) * p.σ * v # nudges v to zero
@inline sponge_w(x, y, z, t, w, p) = -west_mask_cos(x, y, z) * p.σ * w # nudges w to zero
@inline sponge_b(x, y, z, t, b, p) = -west_mask_cos(x, y, z) * p.σ * (b - b∞(z, p)) # nudges b to b∞
#----






#++++ Assembling forcings and BCs
Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
Fw = Forcing(sponge_w, field_dependencies = :w, parameters = params)
Fb = Forcing(sponge_b, field_dependencies = :b, parameters = params)
forcing = (u=Fᵤ, v=Fᵥ, w=Fw, b=Fb)


b_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(0),
                                top = FluxBoundaryCondition(0),
                                west = ValueBoundaryCondition(b_west, parameters=params), # Hidden behind sponge layer
                                east = FluxBoundaryCondition(0), # Hidden behind sponge layer
                                )

u_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(0),
                                top = FluxBoundaryCondition(0),
                                west = OpenBoundaryCondition(u_west, parameters=params),
                                east = OpenBoundaryCondition(u_east, parameters=params),
                                )
w_bcs = FieldBoundaryConditions(west = drag_bc_w)
v_bcs = FieldBoundaryConditions(west = drag_bc_v)
boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs,)
#----


#++++ Construct model
if LES
    closure = SmagorinskyLilly(C=0.16, Pr=0.8, ν=0, κ=0)
else
    closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4)
end
model = NonhydrostaticModel(grid = grid, 
                            advection = WENO5(),
                            timestepper = :RungeKutta3, 
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            coriolis = FPlane(1.2e-4),
                            closure = closure,
                            forcing = forcing,
                            boundary_conditions = boundary_conditions,
                            )
@info "Model" model
#----


#++++ Create simulation
using Oceanostics: SingleLineProgressMessenger
using Oceananigans.Grids: min_Δz

Δt₀ = 1/2 * min_Δz(grid) / (u₁_west + 1e-1)

wizard = TimeStepWizard(cfl=0.9, # How to adjust the time step
                        diffusive_cfl=0.9,
                        max_change=1.02, min_change=0.2, max_Δt=Inf, min_Δt=0.1seconds)

start_time = time_ns() * 1e-9
simulation = Simulation(model, Δt=Δt₀,
                        stop_time = 2days, # when to stop the simulation
)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2)) # When to adjust the time step

# what to print on screen
progress = SingleLineProgressMessenger(SI_units=true,
                                       initial_wall_time_seconds=start_time)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10)) # when to print on screen
@info "Simulation" simulation
#---


#++++ Impose initial conditions
@info "Imposing initial conditions"
b_ic(x, y, z) = b∞(z, params)
set!(model, b=b_ic)
#----


#++++ Outputs
@info "Creating output fields"
u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b        # unpack buoyancy `Field`

# y-component of vorticity
ω_y = Field(∂z(u) - ∂x(w))

outputs = (; u, v, w, b, ω_y)


if mass_flux
    saved_output_prefix = "iceplume"
else
    saved_output_prefix = "iceplume_nomf"
end
saved_output_filename = saved_output_prefix * ".nc"

simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs, 
                                                        schedule = TimeInterval(30minutes),
                                                        filepath = saved_output_filename,
                                                        mode = "c")
#----

#++++ Ready to press the big red button:
run!(simulation)
#----
