--[[
Example on how to relax a geometry using the LBFGS 
algorithm.

This example can take any geometry and will relax it
according to the siesta input options:

 - MD.MaxForceTol
 - MD.MaxCGDispl

One should note that the LBFGS algorithm first converges
when the total force (norm) on the atoms are below the 
tolerance. This is contrary to the SIESTA default which
is a force tolerance for the individual directions,
i.e. max-direction force.

This example is prepared to easily create
a combined relaxation of several LBFGS algorithms
simultaneously. In some cases this is shown to
speed up the convergence because an average is taken
over several optimizations.

To converge using several LBFGS algorithms simultaneously
may be understood phenomenologically by a "line-search" 
optimization by weighing two Hessian optimizations.

This example defaults to two simultaneous LBFGS algorithms
which seems adequate in most situations.

--]]

-- Load the FLOS module
local flos = require "flos"

-- Create the two LBFGS algorithms with
-- initial Hessians 1/75 and 1/50
local LBFGS = {}
LBFGS[1] = flos.LBFGS{H0 = 1. / 75.}
LBFGS[2] = flos.LBFGS{H0 = 1. / 50.}
-- To use more simultaneously simply add a
-- new line... with a separate LBFGS algorithm.

-- Grab the unit table of siesta (it is already created
-- by SIESTA)
local Unit = siesta.Units


function siesta_comm()
   
   -- This routine does exchange of data with SIESTA
   local ret_tbl = {}

   -- Do the actual communication with SIESTA
   if siesta.state == siesta.INITIALIZE then
      
      -- In the initialization step we request the
      -- convergence criteria
      --  MD.MaxDispl
      --  MD.MaxForceTol
      siesta.receive({"MD.MaxDispl",
		      "MD.MaxForceTol"})


      -- Print information (only on IONode)
      IOprint("\nLUA convergence information for the LBFGS algorithms:")

      -- Ensure we update the convergence criteria
      -- from SIESTA (in this way one can ensure siesta options)
      for i = 1, #LBFGS do
	 
	 LBFGS[i].tolerance = siesta.MD.MaxForceTol * Unit.Ang / Unit.eV
	 LBFGS[i].max_dF = siesta.MD.MaxDispl / Unit.Ang

	 -- Print information
	 if siesta.IONode then
	    LBFGS[i]:info()
	 end
      end

   end

   if siesta.state == siesta.MOVE then
      
      -- Here we are doing the actual LBFGS algorithm.
      -- We retrieve the current coordinates, the forces
      -- and whether the geometry has relaxed
      siesta.receive({"geom.xa",
		      "geom.fa",
		      "MD.Relaxed"})
      ret_tbl = siesta_move(siesta)
      
   end

   siesta.send(ret_tbl)
end

function siesta_move(siesta)

   -- Retrieve the atomic coordinates and the forces
   local xa = flos.Array.from(siesta.geom.xa) / Unit.Ang
   local fa = flos.Array.from(siesta.geom.fa) * Unit.Ang / Unit.eV

   -- Perform step (initialize arrays to do averaging if more
   -- LBFGS algorithms are in use).
   local all_xa = {}
   local weight = flos.Array.empty(#LBFGS)
   for i = 1, #LBFGS do
      
      -- Calculate the next optimized structure (that
      -- minimizes the Hessian)
      all_xa[i] = LBFGS[i]:optimize(xa, fa)
      
      -- Get the optimization length for calculating
      -- the best average.
      weight[i] = LBFGS[i].weight
      
   end
   
   -- Normalize weights
   weight = weight / weight:sum()
   if #LBFGS > 1 then
      IOprint("\nLBFGS weighted average: ", weight)
   end

   -- Calculate the new coordinates and figure out
   -- if the algorithms has been optimized.
   local out_xa = all_xa[1] * weight[1]
   local relaxed = LBFGS[1]:optimized()
   for i = 2, #LBFGS do
      
      out_xa = out_xa + all_xa[i] * weight[i]
      relaxed = relaxed and LBFGS[i]:optimized()
      
   end
   -- Immediately clean-up to reduce memory overhead (force GC)
   all_xa = nil

   -- Send back new coordinates (convert to Bohr)
   siesta.geom.xa = out_xa * Unit.Ang
   siesta.MD.Relaxed = relaxed
   
   return {"geom.xa",
	   "MD.Relaxed"}
end
