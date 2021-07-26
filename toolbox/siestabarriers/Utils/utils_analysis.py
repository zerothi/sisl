def NEB_Barrier(NEBfolder,image_number,fig_name,dpi_in ):
    """
    #==========================================================================
    # Script for Energy vs Basis calculation
    # Written by Arsalan Akhtar
    # ICN2 31-August-2017 v-0.1
    # ICN2 1-October-2018 v-0.2
    # ICN2 2-October-2018 v-0.3
    # ICN2 2-October-2018 v-0.4
    #==========================================================================

    """

    # Libray imports
    import os, shutil
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy import interpolate

    print ("*******************************************************************")
    print ("This is a test script written by A.Akhtar for Ploting NEB")
    print ("Email:arsalan.akhtar@icn2.cat")
    print ("*******************************************************************")
    Number_of_images = image_number

    NAME_OF_NEB_RESULTS = NEBfolder + 'NEB.results'
    data=[]
    with open(NAME_OF_NEB_RESULTS) as f:
        for ind, line in enumerate(f, 1):
            if ind>2:
                #print (line)
                data.append(line)
    while '\n' in data:
        data.remove('\n')

    image_number=[]
    reaction_coordinates=[]
    Energy=[]
    E_diff=[]
    Curvature=[]
    Max_Force=[]
    for i in range(len(data)):
        image_number.append(float(data[i].split()[0]))
        reaction_coordinates.append(float(data[i].split()[1]))
        Energy.append(float(data[i].split()[2]))
        E_diff.append(float(data[i].split()[3]))
        Curvature.append(float(data[i].split()[4]))
        Max_Force.append(float(data[i].split()[5]))

    Total_Number=Number_of_images+2
    shift=len(E_diff)-Total_Number


    im=[]
    x=[]
    y=[]
    y2=[]
    for i in range(Total_Number):
        im.append(np.array(int(image_number[shift+i])))
        x.append(np.array(reaction_coordinates[shift+i]))
        y.append(np.array(E_diff[shift+i]))
        y2.append(np.array(Energy[shift+i]))

    #Finding Barrier
    Barrier=max(y)




    xnew = np.linspace(0, x[len(x)-1], num=1000, endpoint=True)
    f1=interp1d(x, y,kind='linear')
    f2=interp1d(x, y, kind='cubic')
    f3=interp1d(x, y, kind='quadratic')


    plt.plot(x,y,"o",xnew,f1(xnew),"-",xnew,f2(xnew),"--",xnew,f3(xnew),'r')
    plt.title("Barrier Energy = "+str(Barrier)+" eV")
    plt.legend(['data', 'linear', 'cubic','quadratic'], loc='best')


    plt.savefig(str(NEBfolder) + str(fig_name) + '.png')
    plt.savefig(str(NEBfolder) + str(fig_name) + '.pdf')
    plt.savefig(str(NEBfolder) + str(fig_name) + '.jpeg',dpi=dpi_in)
    #plt.plot(x,y,"o",x,ynew,'+')
    #if Plot == True:
    #    plt.savefig(inputfile +  fig_name + '.png')
    #    plt.savefig(inputfile +  fig_name + '.pdf')
    #    plt.savefig(inputfile +  fig_name + '.jpeg',dpi=dpi_in)
    #else:
    #    plt.show()
    #return plt


def NEB_Fix (path,fdf_name,xyz_input,xyz_file_out):
    """
    """
    import sisl
    import numpy as np
    fdf = sisl.get_sile(path+fdf_name)
    geom = fdf.read_geometry()
    geom_ase = geom.toASE()
    missing = geom_ase.get_chemical_symbols()

    # %%
    temp_xyz =(path + xyz_input)
    f = open(temp_xyz, "r")

    # %%
    coor = np.array([])
    count = 0
    for i in f:
        if count>1:
            #print (i.split())
            coor = np.append(coor,float(i.split()[0]))
            coor = np.append(coor,float(i.split()[1]))
            coor = np.append(coor,float(i.split()[2]))
        count = count + 1
    #lines = f.readlines()
    f.close()

# %%
    coor = coor.reshape(geom.na,3)

# %%
    #missing_symbols = np.array(['',''])
    missing_symbols = np.array(missing)

    # %%
    #missing_symbols= np.append(missing_symbols, missing)

# %%
    missing_symbols= missing_symbols.reshape(geom.na,1)

# %%
    new_xyz_coor = np.hstack((missing_symbols,coor))

# %%

    f = open(xyz_file_out+".xyz", "w")
    f.writelines(str(geom.na)+'\n')
    f.write(xyz_file_out+'\n')
    for item in new_xyz_coor:
        f.writelines(" {}\t {}  {}  {}\n".format(item[0],item[1],item[2],item[3]))
    f.close()

    print ('Done')



