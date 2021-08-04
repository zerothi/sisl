
def Ghost_block(sisl_images):
    """
    """
    import sisl
    import numpy as np 
    Ghost_Species =np.array([])
    Ghost_Species_name =np.array([])
    count = 0
    n_ghost = 0
    for i in range(sisl_images[0].atoms.nspecie):
        count = int(count+1) 
        print (count , sisl_images[0].atoms.atom[i].Z)
        #print (count)
        if sisl_images[0].atoms.atom[i].Z < 0:
            n_ghost = n_ghost + 1
            a = sisl.Atom(abs(sisl_images[0].atoms.atom[i].Z))
            Ghost_Species = np.append(Ghost_Species,int(count))
            Ghost_Species = np.append(Ghost_Species,int(sisl_images[0].atoms.atom[i].Z))
            Ghost_Species_name = np.append(Ghost_Species_name,a.symbol+'_ghost')
    
    Ghost_Species = Ghost_Species.reshape(n_ghost,2)
    Ghost_Species_name = Ghost_Species_name.reshape(n_ghost,1)

    F = open('ghost_block_temp','w')
    F.writelines('\n')
    F.writelines('%block Geometry-Constraints\n')
    for i in range(Ghost_Species.shape[0]):
        F.writelines("species-i  {:}  \n".format(int(Ghost_Species[i][0])))
    F.writelines('%endblock Geometry-Constraints\n')
    F.writelines('\n')
    F.writelines('%include parameters.fdf \n')
    F.close()


def file_cat (file_name,file_to_added,file_out):
    """
    # Python program to
    # demonstrate merging
    # of two files
    """
    data = data2 = ""
    # Reading data from file1
    with open(file_name) as fp:
        data = fp.read()
    # Reading data from file2
    with open(file_to_added) as fp:
        data2 = fp.read()
    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += data2
    with open (file_out, 'w') as fp:
        fp.write(data)

def string_cat (file_name,string_to_added,file_out):
    """
    # Python program to
    # demonstrate merging
    # of two files
    """
    data = data2 = ""
    # Reading data from file1
    with open(file_name) as fp:
        data = fp.read()
    
    # Reading data from file2
    #with open(file_to_added) as fp:
    #    data2 = fp.read()
    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += string_to_added
    with open (file_out, 'w') as fp:
        fp.write(data)



def replace(textToSearch,textToReplace,fileToSearch):
    import os
    import sys
    import fileinput
    tempFile = open( fileToSearch, 'r+' )
    for line in fileinput.input( fileToSearch ):
        if textToSearch in line :
            print(line,'Match Found')
        #else:
            #print('Match Not Found!!')
        tempFile.write( line.replace( textToSearch, textToReplace ) )
    tempFile.close()
