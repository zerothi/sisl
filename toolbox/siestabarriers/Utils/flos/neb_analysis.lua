--[[
Example on how to use an NEB Analysis.
--]]
--===================================================
-- USER Define Vaiables
-- The prefix of the files that contain the images
local image_label = "images-"
local image_number = 3
local cleanup ="no"
--===================================================
-- Load the FLOS module
local flos = require "flos"
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
images[1]= flos.MDStep{R=read_geom(image_label .. image_number .. ".xyz")}
-- Grab the unit table of siesta (it is already created
-- by SIESTA)
local Unit = siesta.Units


function siesta_comm()
   -- This routine does exchange of data with SIESTA
   --local ret_tbl = {}
     -- Do the actual communication with SIESTA
  if siesta.state == siesta.INITIALIZE then  --INITIALIZE
      siesta.receive({"Label","geom.xa","MD.Relaxed"})
      --siesta.receive({"MD.Relaxed"})
      --Store the Label
      label = tostring(siesta.Label)
      -- Print information
      siesta_update_DM(image_number)
      IOprint("\nLUA NEB Analysis for image :".. image_label .. "calculator")  
      -- This is only reached one time, and that it as the beginning...
      -- be sure to set the corresponding values
      siesta.geom.xa = images[1].R* Unit.Ang  --NEB.initial.R * Unit.Ang
      siesta.send({"geom.xa"})
      IOprint("\nLUA NEB Analysis for image Coordinates :\n")
      IOprint(images[1].R* Unit.Ang )

  end

  if siesta.state == siesta.MOVE then
      -- Here we are doing the actual LBFGS algorithm.
      -- We retrieve the current coordinates, the forces
      -- and whether the geometry has relaxed
      siesta.receive({"MD.Relaxed"})
      IOprint("\nLUA/NEB MOVE state STARTED\n")
      siesta.MD.Relaxed = true
      siesta.send({"MD.Relaxed"})
  end
  
    if siesta.state == siesta.ANALYSIS_AFTER then
      IOprint("\nLUA/NEB AFTER ANALYSIS state STARTED\n")
      siesta_update_Analysis(image_number)
     --os.execute("cp " .. label .. ".PDOS" .. " " .. label .. ".PDOS.3")
   --   IOprint("\nLUA/NEB MOVE state FINISHED\n")

      siesta_cleanup(cleanup)
      --ret_tbl = siesta_move(siesta)
  end
   
end
 
function siesta_update_DM(image_number)

   if not siesta.IONode then
      -- only allow the IOnode to perform stuff...
      return
   end
   -- Move about files so that we re-use old DM files
   local DM = label .. ".DM"
   local current_DM = DM .. "." .. tostring(image_number)  
   -- Saving initial DM
   IOprint("Saving " .. current_DM .. " to " .. DM)
   os.execute("cp " .. current_DM .. " " .. DM)   
   
end

function siesta_update_Analysis(image_number)

   if not siesta.IONode then
      -- only allow the IOnode to perform stuff...
      return
   end
   -- Move about files so that we re-use old DM files
   local PDOS = label .. ".PDOS"
   local DOS = label .. ".DOS"
   local EIG = label .. ".EIG"
   local BAND = label .. ".bands"
   local FA = label .. ".FA"
   local FAC = label .. ".FAC"
   local PDOSKP = label .. ".PDOS.KP"
   local PDOSxml = label .. ".PDOS.xml"
   local XV = label .. ".XV"
   local XYZ = label .. ".xyz"
   local FORCE_STRESS = "FORCE_STRESS"
   local current_PDOS = PDOS .. "." .. tostring(image_number)
   local current_DOS = DOS .. "." .. tostring(image_number)
   local current_EIG = EIG .. "." .. tostring(image_number) 
   local current_BAND = BAND .. "." .. tostring(image_number) 
   local current_FA = FA .. "." .. tostring(image_number) 
   local current_FAC = FAC .. "." .. tostring(image_number) 
   local current_PDOSKP = PDOSKP .. "." .. tostring(image_number) 
   local current_PDOSxml = PDOSxml .. "." .. tostring(image_number) 
   local current_FORCE_STRESS = FORCE_STRESS .. "." .. tostring(image_number) 
   local current_XV = XV .. "." .. tostring(image_number) 
   local current_XYZ = XYZ .. "." .. tostring(image_number) 
   -- Saving initial PDOS
   IOprint("Saving " .. PDOS .. " to " .. current_PDOS)
   os.execute("mv " .. PDOS  .. " " .. current_PDOS) 
   -- Saving initial DOS
   IOprint("Saving " .. DOS  .. " to " .. current_DOS)
   os.execute("mv " .. DOS  .. " " .. current_DOS)  
   -- Saving initial EIG
   IOprint("Saving " .. EIG  .. " to " .. current_EIG)
   os.execute("mv " .. EIG  .. " " .. current_EIG)
   -- Saving initial BAND
   IOprint("Saving " .. BAND .. " to " ..  current_BAND)
   os.execute("mv " .. BAND  .. " " .. current_BAND)   
   -- Saving initial FA
   IOprint("Saving " .. FA .. " to " ..  current_FA)
   os.execute("mv " .. FA  .. " " .. current_FA)
  -- Saving initial FAC
   IOprint("Saving " .. FAC .. " to " ..  current_FAC)
   os.execute("mv " .. FAC  .. " " .. current_FAC)
   -- Saving initial PDOSKP
   IOprint("Saving " .. PDOSKP .. " to " ..  current_PDOSKP)
   os.execute("mv " .. PDOSKP  .. " " .. current_PDOSKP)
   -- Saving initial PDOSxml
   IOprint("Saving " .. PDOSxml .. " to " ..  current_PDOSxml)
   os.execute("mv " .. PDOSxml  .. " " .. current_PDOSxml)
    -- Saving initial XV
   IOprint("Saving " .. XV .. " to " ..  current_XV)
   os.execute("mv " .. XV  .. " " .. current_XV)
    -- Saving initial xyz
   IOprint("Saving " .. XYZ .. " to " ..  current_XYZ)
   os.execute("mv " .. XYZ  .. " " .. current_XYZ)
   
   -- Saving initial FORCE_STRESS
   IOprint("Saving " .. FORCE_STRESS .. " to " ..  current_FORCE_STRESS)
   os.execute("mv " .. FORCE_STRESS  .. " " .. current_FORCE_STRESS)
   
end

function siesta_cleanup(yes)

   if not siesta.IONode then
      -- only allow the IOnode to perform stuff...
      return
   end
   -- Move about files so that we re-use old DM files
   -- Saving initial PDOS
   --IOprint("Saving " .. PDOS .. " to " .. current_PDOS)
   if yes=="yes" then
     IOprint("These Files Ghoing to be Deleted :\n")
     IOprint("BASIS_ENTHALPY\n" ..
            "chlocal.charge\n" ..
            "PARALLEL_DIST\n"..
            "NON_TRIMMED_KP_LIST\n"..
            "*.log\n" ..
            "CHLOCAL.*\n".. 
            "*.BONDS* \n"..
            "INPUT_TMP.* \n"..
            "ORB.* \n"..
            "SPLIT_SCAN*\n".. 
            "RED_VLOCAL.* \n"..
            "*.Vxc* \n"..
            "*.Vhart*\n" ..
            "*.psdump\n" ..
            "*.ion.*\n" ..
            "VNA.*\n" ..
            "*.confpot\n" .. 
            "*.charge\n" ..
            "KB.*\n"..
            "*STRUCT_OUT*\n"..
            "*.Vsoft\n"
            )
      os.execute("rm BASIS_ENTHALPY BASIS_HARRIS_ENTHALPY chlocal.charge PARALLEL_DIST NON_TRIMMED_KP_LIST ")
   os.execute("rm *.log CHLOCAL.* *.BONDS* INPUT_TMP.* ORB.* SPLIT_SCAN* RED_VLOCAL.* *.Vxc* *.Vhart* *.psdump *.ion.* VNA.* *.confpot *.charge KB.* *STRUCT_OUT* *.Vsoft " )
   end
end
