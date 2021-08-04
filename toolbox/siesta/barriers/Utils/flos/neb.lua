--[[
Example on how to use an NEB method.
--]]
--=================================================================
local image_label = "images-"
-- Total number of images (excluding initial[0] and final[n_images+1])
local n_images = 7
local k_spring = 0.1
--=================================================================
-- Load the FLOS module
local flos = require "flos"
-- The prefix of the files that contain the images
-- Table of image geometries
local images = {}

-- The default output label of the DM files
-- Function for reading a geometry
local read_geom = function(filename)
   local file = io.open(filename, "r")
   local na = tonumber(file:read())
   local R = flos.Array.zeros(na, 3)
   file:read()
   local i = 0
   local function tovector(s)
      local t = {}
      s:gsub('%S+', function(n) t[#t+1] = tonumber(n) end)
      return t
   end
   for i = 1, na do
      local line = file:read()
      if line == nil then break end
      -- Get stuff into the R
      local v = tovector(line)
      R[i][1] = v[1]
      R[i][2] = v[2]
      R[i][3] = v[3]
   end
   file:close()
   return R
end

-- Now read in the images
for i = 0, n_images + 1 do
   images[#images+1] = flos.MDStep{R=read_geom(image_label .. i .. ".xyz")}
end

-- Now we have all images...
local NEB = flos.NEB(images,{k=k_spring})
if siesta.IONode then
   NEB:info()
end
-- Remove global (we use NEB.n_images)
n_images = nil

-- Setup each image relaxation method (note it is prepared for several
-- relaxation methods per-image)
local relax = {}
for i = 1, NEB.n_images do
   -- Select the relaxation method
   relax[i] = {}
   relax[i][1] = flos.CG{beta='PR',restart='Powell', line=flos.Line{optimizer = flos.LBFGS{H0 = 1. / 25.} } }
     -- add more relaxation schemes if needed ;)
   --relax[i][2] = flos.CG{beta='PR',restart='Powell', line=flos.Line{optimizer = flos.LBFGS{H0 = 1. / 50.} } }
   --relax[i][3] = flos.CG{beta='PR',restart='Powell', line=flos.Line{optimizer = flos.LBFGS{H0 = 1. / 75.} } }
   --relax[i][4] = flos.LBFGS{H0 = 1. / 50}
   --relax[i][5] = flos.FIRE{dt_init = 1., direction="global", correct="global"}
   if siesta.IONode then
      NEB:info()
    end

end

-- Counter for controlling which image we are currently relaxing
local current_image = 1

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
      siesta.receive({"Label",
		      "geom.xa",
		      "MD.MaxDispl",
		      "MD.MaxForceTol"})

      -- Store the Label
      label = tostring(siesta.Label)

      -- Print information
      IOprint("\nLUA NEB calculator")

      -- Ensure we update the convergence criteria
      -- from SIESTA (in this way one can ensure siesta options)
      for img = 1, NEB.n_images do
	 IOprint(("\nLUA NEB relaxation method for image %d:"):format(img))
	 for i = 1, #relax[img] do
	    relax[img][i].tolerance = siesta.MD.MaxForceTol * Unit.Ang / Unit.eV
	    relax[img][i].max_dF = siesta.MD.MaxDispl / Unit.Ang
	    
	    -- Print information for this relaxation method
	    if siesta.IONode then
	       relax[img][i]:info()
	    end
	 end
      end

      -- This is only reached one time, and that it as the beginning...
      -- be sure to set the corresponding values
      siesta.geom.xa = NEB.initial.R * Unit.Ang

      IOprint("\nLUA/NEB initial state\n")
      -- force the initial image to be the first one to run
      current_image = 0
      siesta_update_DM(0, current_image)
      --Write xyz File
      siesta_update_xyz(current_image)
      IOprint(NEB[current_image].R)

      ret_tbl = {'geom.xa'}

   end

   if siesta.state == siesta.MOVE then
      
      -- Here we are doing the actual LBFGS algorithm.
      -- We retrieve the current coordinates, the forces
      -- and whether the geometry has relaxed
      siesta.receive({"geom.fa",
		      "E.total",
		      "MD.Relaxed"})

      -- Store the old image that has been tested,
      -- in this way we can check whether we have moved to
      -- a new image.
      local old_image = current_image
      
      ret_tbl = siesta_move(siesta)

      -- we need to re-organize the DM files for faster convergence
      -- pass whether the image is the same
      siesta_update_DM(old_image, current_image)
      siesta_update_xyz(current_image)
      IOprint(NEB[current_image].R)

   end

   siesta.send(ret_tbl)
end

function siesta_move(siesta)

   -- Retrieve the atomic coordinates, forces and the energy
   local fa = flos.Array.from(siesta.geom.fa) * Unit.Ang / Unit.eV
   local E = siesta.E.total / Unit.eV

   -- First update the coordinates, forces and energy for the
   -- just calculated image
   NEB[current_image]:set{F=fa, E=E}

   if current_image == 0 then
      -- Perform the final image, to retain that information
      current_image = NEB.n_images + 1

      -- Set the atomic coordinates for the final image
      siesta.geom.xa = NEB[current_image].R * Unit.Ang

      IOprint("\nLUA/NEB final state\n")
      --siesta_update_DM(0, current_image)
      -- The siesta relaxation is already not set
      return {'geom.xa'}
      
   elseif current_image == NEB.n_images + 1 then

      -- Start the NEB calculation
      current_image = 1

      -- Set the atomic coordinates for the final image
      siesta.geom.xa = NEB[current_image].R * Unit.Ang

      IOprint(("\nLUA/NEB running NEB image %d / %d\n"):format(current_image, NEB.n_images))
	 
      -- The siesta relaxation is already not set
      return {'geom.xa'}

   elseif current_image < NEB.n_images then

      current_image = current_image + 1

      -- Set the atomic coordinates for the image
      siesta.geom.xa = NEB[current_image].R * Unit.Ang
      
      IOprint(("\nLUA/NEB running NEB image %d / %d\n"):format(current_image, NEB.n_images))
      
      -- The siesta relaxation is already not set
      return {'geom.xa'}

   end
   
   -- First we figure out how perform the NEB optimizations
   -- Now we have calculated all the systems and are ready for doing
   -- an NEB MD step

   -- Global variable to check for the NEB convergence
   -- Initially assume it has relaxed
   local relaxed = true

   IOprint("\nNEB step")
   local out_R = {}

   -- loop on all images and pass the updated forces to the mixing algorithm
   for img = 1, NEB.n_images do

      -- Get the correct NEB force (note that the relaxation
      -- methods require the negative force)
      local F = NEB:force(img, siesta.IONode)
      IOprint("NEB: max F on image ".. img ..
		 (" = %10.5f, climbing = %s"):format(F:norm():max(),
						     tostring(NEB:climbing(img))) )

      -- Prepare the relaxation for image `img`
      local all_xa, weight = {}, flos.Array( #relax[img] )
      for i = 1, #relax[img] do
	 all_xa[i] = relax[img][i]:optimize(NEB[img].R, F)
	 weight[i] = relax[img][i].weight
      end
      weight = weight / weight:sum()

      if #relax[img] > 1 then
	 IOprint("\n weighted average for relaxation: ", tostring(weight))
      end
      
      -- Calculate the new coordinates and figure out
      -- if the algorithm has converged (all forces below)
      local out_xa = all_xa[1] * weight[1]
      relaxed = relaxed and relax[img][1]:optimized()
      for i = 2, #relax[img] do
	 out_xa = out_xa + all_xa[i] * weight[i]
	 relaxed = relaxed and relax[img][i]:optimized()
      end
      
      -- Copy the optimized coordinates to a table
      out_R[img] = out_xa

   end

   -- Before we update the coordinates we will write
   -- the current steps results to the result file
   -- (this HAS to be done before updating the coordinates)
   NEB:save( siesta.IONode )

   -- Now we may copy over the coordinates (otherwise
   -- we do a consecutive update, and then overwrite)
   for img = 1, NEB.n_images do
      NEB[img]:set{R=out_R[img]}
   end
   
   -- Start over in case the system has not relaxed
   current_image = 1
   if relaxed then
      -- the final coordinates are returned
      siesta.geom.xa = NEB.final.R * Unit.Ang
      IOprint("\nLUA/NEB complete\n")
   else
      siesta.geom.xa = NEB[1].R * Unit.Ang
      IOprint(("\nLUA/NEB running NEB image %d / %d\n"):format(current_image, NEB.n_images))
   end

   siesta.MD.Relaxed = relaxed
      
   return {"geom.xa",
	   "MD.Relaxed"}
end

-- Function for retaining the DM files for the images so that we
-- can easily restart etc.
function siesta_update_DM(old, current)

   if not siesta.IONode then
      -- only allow the IOnode to perform stuff...
      return
   end
   -- Move about files so that we re-use old DM files
   local DM = label .. ".DM"
   local old_DM = DM .. "." .. tostring(old)
   local current_DM = DM .. "." .. tostring(current)
   local initial_DM = DM .. ".0"
   local final_DM = DM .. ".".. tostring(NEB.n_images+1) 
   print ("The Label of Old DM is : " .. old_DM)
   print ("The Label of Current DM is : " .. current_DM)
   -- Saving initial DM
   if old==0 and current==0 then
     print("Removing DM for Resuming")
     IOprint("Deleting " .. DM .. " for a clean restart...")
     os.execute("rm " .. DM)
   end 
  
   if 0 <= old and old <= NEB.n_images+1 and NEB:file_exists(DM) then
      -- store the current DM for restart purposes
      IOprint("Saving " .. DM .. " to " .. old_DM)
      os.execute("mv " .. DM .. " " .. old_DM)
   elseif NEB:file_exists(DM) then
      IOprint("Deleting " .. DM .. " for a clean restart...")
      os.execute("rm " .. DM)
   end

   if NEB:file_exists(current_DM) then
      IOprint("Deleting " .. DM .. " for a clean restart...")
      os.execute("rm " .. DM)
      IOprint("Restoring " .. current_DM .. " to " .. DM)
      os.execute("cp " .. current_DM .. " " .. DM)
   end

end

function siesta_update_xyz(current)
  if not siesta.IONode then
      -- only allow the IOnode to perform stuff...
      return
   end
  local xyz_label = image_label ..tostring(current)..".xyz"
  --self:_n_images=self.n_images
  --self:_check_image(image)
    
  local f=io.open(xyz_label,"w")
  f:write(tostring(#NEB[current].R).."\n \n")
  --f:write(tostring(initialize:self.n_images).."\n \n")
  for i=1,#NEB[current].R do
    f:write(string.format(" %19.17f",tostring(NEB[current].R[i][1])).. "   "..string.format("%19.17f",tostring(NEB[current].R[i][2]))..string.format("   %19.17f",tostring(NEB[current].R[i][3])).."\n")
 end
 f:close()
  --  
end
