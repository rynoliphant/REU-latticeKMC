import numpy as np
from random import shuffle
import time

class Element:
    elements = ['X','Ga','N','H','Si','O']
    mass = [0,69.723,14.0067,1.00784,28.0855,15.999]

    def __init__(self,elemnt:str,elements=elements,mass=mass):
        self.e = elemnt

        try:
            idx=elements.index(elemnt)
        except:
            print("ERROR:",elemnt, "is not in the list of elements:", elements)
            self.m = 0
        else:
            idx=elements.index(elemnt)
            self.m=mass[idx]

class Atom:
    def __init__(self, element:Element, ID):
        self.e = element.e
        self.m = element.m
        self.id = ID
        
class Config:
    structures = ['fcc', 'bcc','hcp','zincblende','wurtzite']
    def generate_supercell(self, unit_cell, nx:int, ny:int, nz:int):
        '''
        '''
        start_time = time.time()
        direct_list = [[((i)+site[0])/nx, ((j)+site[1])/ny, ((k)+site[2])/nz] for k in range(nz) 
                       for j in range(ny) for i in range(nx) for site in unit_cell]

        print("Generate Supercell--- %s seconds ---" % (time.time() - start_time))
        return np.array(direct_list)
    
    def n_nearest_neighbor(self, positions, neighbor_dist:list):
        '''
        '''
        start_time = time.time()

        n_neighbor = []
        first_neigh = []
        second_neigh = []
        third_neigh = []
        fourth_neigh = []
        for coor in positions:
            dist = np.linalg.norm(positions-coor,axis=1) #np.array([coor]).T
            first_bool = np.around(dist,5) == round(neighbor_dist[0],5)
            second_bool = np.around(dist,5) == round(neighbor_dist[1],5)
            third_bool = np.around(dist,5) == round(neighbor_dist[2],5)
            fourth_bool = np.around(dist,5) == round(neighbor_dist[3],5)
            first_neigh.append(positions[first_bool])
            second_neigh.append(positions[second_bool])
            third_neigh.append(positions[third_bool])
            fourth_neigh.append(positions[fourth_bool])
        
        #n_neighbor.append(first_neigh)
        #n_neighbor.append(second_neigh)
        #n_neighbor.append(third_neigh)
        #n_neighbor.append(fourth_neigh)

        #pos = np.repeat(np.array([positions]), len(positions),axis=0)
        #trans_pos = np.repeat(np.reshape(np.array([positions]),(len(positions),1,3)), len(positions),axis=1)

        #dist = np.linalg.norm(pos-trans_pos,axis=2)
        #for i,r in enumerate(positions):
        #    first_neigh.append(pos[i][np.around(dist[i],5)== round(neighbor_dist[0],5)])
        #    second_neigh.append(pos[i][np.around(dist[i],5)== round(neighbor_dist[1],5)])
        #    third_neigh.append(pos[i][np.around(dist[i],5)== round(neighbor_dist[2],5)])
        #    fourth_neigh.append(pos[i][np.around(dist[i],5)== round(neighbor_dist[3],5)])
        n_neighbor.append(first_neigh)
        n_neighbor.append(second_neigh)
        n_neighbor.append(third_neigh)
        n_neighbor.append(fourth_neigh)

        print("Nearest Neighbor--- %s seconds ---" % (time.time() - start_time))
        return n_neighbor

    def cartesian_coor(self, positions, a,b,c, alpha, beta, gamma,nx,ny,nz):
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
        cosa = np.cos(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        cosg = np.cos(gamma)
        sing = np.sin(gamma)
        n_2 = (cosa - cosg*cosb)/sing
        conversion_matrix = np.array([[a,0,0],
                                      [b*cosg, b*sing,0],
                                      [c*cosb, c*n_2, c*np.sqrt(sinb**2 - n_2**2)]])
        conversion_matrix[0]=conversion_matrix[0]*nx
        conversion_matrix[1]=conversion_matrix[1]*ny
        conversion_matrix[2]=conversion_matrix[2]*nz
        cartesian = np.matmul(positions,conversion_matrix)

        return cartesian

    def __init__(self, struct:str, lattice_a:float, atom_types:list, ratio:list, inter_atom_types:list, inter_ratio:list,
                  nx=1, ny=1, nz=1, randm=False, structures=structures):
        '''
        '''
        self.struct = struct
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.a = lattice_a
        self.atom_types = atom_types
        self.ratio = ratio
        self.inter_atom_types = inter_atom_types
        self.inter_ratio = inter_ratio

        element_types=[Element(elmnt) for elmnt in atom_types]
        inter_element_types=[Element(inter) for inter in inter_atom_types]

        if struct=='fcc': #-----------------------------------------------------------------------------------------------------
            self.basis_vectors=np.array([[lattice_a,0,0],
                                         [0,lattice_a,0],
                                         [0,0,lattice_a]])
            
            #Lattice Sites
            self.unit_cell=np.array([[0,0,0],
                                     [0.5,0.5,0],
                                     [0.5,0,0.5],
                                     [0,0.5,0.5]])
            self.frac_lat_positions=self.generate_supercell(self.unit_cell,nx,ny,nz)
            self.lat_positions=self.cartesian_coor(self.frac_lat_positions,lattice_a,lattice_a,lattice_a,90,90,90,nx,ny,nz)

            #List of Atoms for Lattice Sites
            total = nx*ny*nz*len(self.unit_cell)
            self.total = total
            atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i,r in enumerate(ratio) 
                         for position in range(round((r/sum(ratio))*total))]
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                shuffle(atom_list)
                self.atoms=atom_list
            else:
                self.atoms = atom_list

            #Interstitial Sites
            self.octa_unit_cell = [[0.5,0.5,0.5],
                                   [0.5,0,0],
                                   [0,0.5,0],
                                   [0,0,0.5]]
            self.tetra_unit_cell = [[0.25,0.25,0.25],
                                    [0.75,0.25,0.25],
                                    [0.25,0.75,0.25],
                                    [0.25,0.25,0.75],
                                    [0.25,0.75,0.75],
                                    [0.75,0.25,0.75],
                                    [0.75,0.75,0.25],
                                    [0.75,0.75,0.75]]
            self.inter_unit_cell = self.octa_unit_cell + self.tetra_unit_cell
            self.frac_inter_positions = self.generate_supercell(self.inter_unit_cell,nx,ny,nz)
            self.inter_positions = self.cartesian_coor(self.frac_inter_positions,lattice_a,lattice_a,lattice_a,90,90,90,nx,ny,nz)

            #List of Atoms for Interstitial Sites
            inter_total = nx*ny*nz*len(self.inter_unit_cell)
            self.inter_total = inter_total
            inter_atom_list = [Atom(inter_element_types[inx],'INT_'+inter_element_types[inx].e+str(pos)) for inx,ra in enumerate(inter_ratio) 
                         for pos in range(round((ra/sum(inter_ratio))*inter_total))]
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.nearest_neighbor = self.n_nearest_neighbor(self.lat_positions, [(lattice_a/np.sqrt(2)),lattice_a,(np.sqrt(3/2)*lattice_a),
                                                                                 (np.sqrt(2)*lattice_a)])#,(sqrt(3)*lattice_a)])
        elif struct=='bcc': #---------------------------------------------------------------------------------------------------
            self.basis_vectors=np.array([[lattice_a,0,0],
                                         [0,lattice_a,0],
                                         [0,0,lattice_a]])
            
            #Lattice Sites
            self.unit_cell=np.array([[0,0,0],
                                     [0.5,0.5,0.5]])
            self.frac_lat_positions=self.generate_supercell(self.unit_cell,nx,ny,nz)
            self.lat_positions=self.cartesian_coor(self.frac_lat_positions,lattice_a,lattice_a,lattice_a,90,90,90,nx,ny,nz)

            #List of Atoms for Lattice Sites
            total = nx*ny*nz*len(self.unit_cell)
            self.total = total
            atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i,r in enumerate(ratio) 
                         for position in range(round((r/sum(ratio))*total))]
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                shuffle(atom_list)
                self.atoms=atom_list
            else:
                self.atoms = atom_list

            #Interstitial Sites
            self.octa_unit_cell = [[0,0.5,0.5],
                                   [0.5,0,0.5],
                                   [0.5,0.5,0],
                                   [0.5,0,0],
                                   [0,0.5,0],
                                   [0,0,0.5]]
            self.tetra_unit_cell = [[0.5,0.25,0],
                                    [0.25,0.5,0],
                                    [0.5,0.75,0],
                                    [0.75,0.5,0],

                                    [0.5,0,0.25],
                                    [0.25,0,0.5],
                                    [0.5,0,0.75],
                                    [0.75,0,0.5],
                                    
                                    [0,0.5,0.25],
                                    [0,0.25,0.5],
                                    [0,0.5,0.75],
                                    [0,0.75,0.5]]
            self.inter_unit_cell = self.octa_unit_cell + self.tetra_unit_cell
            self.frac_inter_positions = self.generate_supercell(self.inter_unit_cell,nx,ny,nz)
            self.inter_positions = self.cartesian_coor(self.frac_inter_positions,lattice_a,lattice_a,lattice_a,90,90,90,nx,ny,nz)

            #List of Atoms for Interstitial Sites
            inter_total = nx*ny*nz*len(self.inter_unit_cell)
            self.inter_total = inter_total
            inter_atom_list = [Atom(inter_element_types[inx],'INT_'+inter_element_types[inx].e+str(pos)) for inx,ra in enumerate(inter_ratio) 
                         for pos in range(round((ra/sum(inter_ratio))*inter_total))]
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.nearest_neighbor = self.n_nearest_neighbor(self.lat_positions, [(np.sqrt(3)*lattice_a*0.5),lattice_a,
                                                                                 (np.sqrt(2)*lattice_a),(np.sqrt(3)*lattice_a)])
        elif struct=='hcp': #---------------------------------------------------------------------------------------------------
            self.basis_vectors=np.array([[lattice_a, 0, 0],
                                         [lattice_a*-0.5, lattice_a*0.866025, 0],
                                         [0, 0, np.sqrt(8/3)*lattice_a]])
            
            #Lattice Sites
            self.unit_cell=np.array([[(2/3), (1/3), 0.5],
                                     [(1/3), (2/3), 1]])
            self.frac_lat_positions=self.generate_supercell(self.unit_cell,nx,ny,nz)
            self.lat_positions=self.cartesian_coor(self.frac_lat_positions,lattice_a,lattice_a,np.sqrt(8/3)*lattice_a,90,90,120,nx,ny,nz)

            #List of Atoms for Lattice Sites
            total = nx*ny*nz*len(self.unit_cell)
            self.total = total
            atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i,r in enumerate(ratio) 
                         for position in range(round((r/sum(ratio))*total))]
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                shuffle(atom_list)
                self.atoms=atom_list
            else:
                self.atoms = atom_list

            #Interstitial Sites
            self.octa_unit_cell = [[1,0,0.25],
                                   [1,0,0.75]]

            self.tetra_unit_cell = [[(1/3),(2/3),(3/8)],
                                    [(1/3),(2/3),(5/8)],
                                    [(2/3),(1/3),(1/8)],
                                    [(2/3),(1/3),(7/8)]]
            self.inter_unit_cell = self.octa_unit_cell + self.tetra_unit_cell
            self.frac_inter_positions = self.generate_supercell(self.inter_unit_cell,nx,ny,nz)
            self.inter_positions=self.cartesian_coor(self.frac_inter_positions,lattice_a,lattice_a,np.sqrt(8/3)*lattice_a,90,90,120,nx,ny,nz)

            #List of Atoms for Interstitial Sites
            inter_total = nx*ny*nz*len(self.inter_unit_cell)
            self.inter_total = inter_total
            inter_atom_list = [Atom(inter_element_types[inx],'INT_'+inter_element_types[inx].e+str(pos)) for inx,ra in enumerate(inter_ratio) 
                         for pos in range(round((ra/sum(inter_ratio))*inter_total))]
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.nearest_neighbor = self.n_nearest_neighbor(self.lat_positions, [lattice_a,(np.sqrt(2)*lattice_a),
                                                                                 np.sqrt(8/3)*lattice_a,(np.sqrt(3)*lattice_a)])
        elif struct=='zincblende': #--------------------------------------------------------------------------------------------
            self.basis_vectors=np.array([[lattice_a,0,0],
                                         [0,lattice_a,0],
                                         [0,0,lattice_a]])
            
            #Lattice Sites
            self.unit_cell=np.array([[0,0,0],
                                     [0.5,0.5,0],
                                     [0.5,0,0.5],
                                     [0,0.5,0.5],
                                     [0.25,0.25,0.25],
                                     [0.75,0.75,0.25],
                                     [0.25,0.75,0.75],
                                     [0.75,0.25,0.75]])
            fcc_lat_positions=self.generate_supercell(self.unit_cell[0:4],nx,ny,nz)
            other_lat_positions=self.generate_supercell(self.unit_cell[4:],nx,ny,nz)
            self.frac_lat_positions=np.array(list(fcc_lat_positions)+list(other_lat_positions))
            self.lat_positions=self.cartesian_coor(self.frac_lat_positions,lattice_a,lattice_a,lattice_a,90,90,90,nx,ny,nz)

            #List of Atoms for Lattice Sites
            total = nx*ny*nz*len(self.unit_cell)
            self.total = total
            atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i,r in enumerate(ratio) 
                         for position in range(round((r/sum(ratio))*total))]
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                #shuffle(atom_list)
                self.atoms=atom_list
            else:
                self.atoms = atom_list

            #Interstitial Sites
            self.octa_unit_cell = [[0.5,0.5,0.5],
                                   [0.5,0,0],
                                   [0,0.5,0],
                                   [0,0,0.5]]
            self.tetra_unit_cell = [[0.75,0.25,0.25],
                                    [0.25,0.75,0.25],
                                    [0.25,0.25,0.75],
                                    [0.75,0.75,0.75]]
            self.inter_unit_cell = self.octa_unit_cell + self.tetra_unit_cell
            self.frac_inter_positions = self.generate_supercell(self.inter_unit_cell,nx,ny,nz)
            self.inter_positions = self.cartesian_coor(self.frac_inter_positions,lattice_a,lattice_a,lattice_a,90,90,90,nx,ny,nz)

            #List of Atoms for Interstitial Sites
            inter_total = nx*ny*nz*len(self.inter_unit_cell)
            self.inter_total = inter_total
            inter_atom_list = [Atom(inter_element_types[inx],'INT_'+inter_element_types[inx].e+str(pos)) for inx,ra in enumerate(inter_ratio) 
                         for pos in range(round((ra/sum(inter_ratio))*inter_total))]
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.nearest_neighbor = self.n_nearest_neighbor(self.lat_positions, [(lattice_a*np.sqrt(3)/4),(lattice_a/np.sqrt(2)),
                                                                                 lattice_a,(np.sqrt(19)*lattice_a/4)])
        elif struct=='wurtzite': #----------------------------------------------------------------------------------------------
            self.basis_vectors=np.array([[lattice_a, 0, 0],
                                         [lattice_a*-0.5, lattice_a*0.866025, 0],
                                         [0, 0, np.sqrt(8/3)*lattice_a]])
            
            #Lattice Sites
            self.unit_cell=np.array([[(2/3), (1/3), 0.5],
                                     [(1/3), (2/3), 0],
                                     [(2/3), (1/3), (7/8)],
                                     [(1/3), (2/3), (3/8)]])
            hcp_lat_positions=self.generate_supercell(self.unit_cell[0:2],nx,ny,nz)
            other_lat_positions=self.generate_supercell(self.unit_cell[2:],nx,ny,nz)
            self.frac_lat_positions=np.array(list(hcp_lat_positions)+list(other_lat_positions))
            self.lat_positions=self.cartesian_coor(self.frac_lat_positions,lattice_a,lattice_a,np.sqrt(8/3)*lattice_a,90,90,120,nx,ny,nz)

            #List of Atoms for Lattice Sites
            total = nx*ny*nz*len(self.unit_cell)
            self.total = total
            atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i,r in enumerate(ratio) 
                         for position in range(round((r/sum(ratio))*total))]
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                #shuffle(atom_list)
                self.atoms=atom_list
            else:
                self.atoms = atom_list

            #Interstitial Sites
            self.octa_unit_cell = [[1,0,0.25],
                                   [1,0,0.75]]

            self.tetra_unit_cell = [[(1/3),(2/3),(5/8)],
                                    [(2/3),(1/3),(1/8)]]
            self.inter_unit_cell = self.octa_unit_cell + self.tetra_unit_cell
            self.frac_inter_positions = self.generate_supercell(self.inter_unit_cell,nx,ny,nz)
            self.inter_positions=self.cartesian_coor(self.frac_inter_positions,lattice_a,lattice_a,np.sqrt(8/3)*lattice_a,90,90,120,nx,ny,nz)

            #List of Atoms for Interstitial Sites
            inter_total = nx*ny*nz*len(self.inter_unit_cell)
            self.inter_total = inter_total
            inter_atom_list = [Atom(inter_element_types[inx],'INT_'+inter_element_types[inx].e+str(pos)) for inx,ra in enumerate(inter_ratio) 
                         for pos in range(round((ra/sum(inter_ratio))*inter_total))]
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.nearest_neighbor = self.n_nearest_neighbor(self.lat_positions, [np.sqrt(3/8)*lattice_a,lattice_a,
                                                                                 (5/(2*np.sqrt(6)))*lattice_a,np.sqrt(11/8)*lattice_a])
                                                                                 #np.sqrt(2)*lattice_a,np.sqrt(8/3)*lattice_a])
        else: #-----------------------------------------------------------------------------------------------------------------
            print('ERROR:',struct,"is not an available structure:",structures)
            self.basis_vectors=np.array([[0,0,0],[0,0,0],[0,0,0]])

def POSCAR_saveFile (output_file,lattice:Config, cartesian=False, show_inter=False):
    start_time = time.time()
    with open(output_file, 'w') as f:
        f.write(lattice.struct+' structure\n')
        f.write('1.0\n')
        f.write(f'{lattice.basis_vectors[0][0]*lattice.nx} {lattice.basis_vectors[0][1]*lattice.nx} {lattice.basis_vectors[0][2]*lattice.nx}\n')
        f.write(f'{lattice.basis_vectors[1][0]*lattice.ny} {lattice.basis_vectors[1][1]*lattice.ny} {lattice.basis_vectors[1][2]*lattice.ny}\n')
        f.write(f'{lattice.basis_vectors[2][0]*lattice.nz} {lattice.basis_vectors[2][1]*lattice.nz} {lattice.basis_vectors[2][2]*lattice.nz}\n')
        if cartesian==False:
            order=[]
            for element in lattice.atom_types:
                f.write(f'{element} ')
                ele_bool = [obj.e==element for obj in lattice.atoms]
                order=order+list(lattice.frac_lat_positions[ele_bool])
            inter_order=[]
            if show_inter==True:
                for elemnt in lattice.inter_atom_types:
                    f.write(f'{elemnt} ')
                    inter_bool = [objt.e==elemnt for objt in lattice.inter_atoms]
                    inter_order=inter_order+list(lattice.frac_inter_positions[inter_bool])
        else:
            order=[]
            for element in lattice.atom_types:
                f.write(f'{element} ')
                ele_bool = [obj.e==element for obj in lattice.atoms]
                order=order+list(lattice.lat_positions[ele_bool])
            inter_order=[]
            if show_inter==True:
                for elemnt in lattice.inter_atom_types:
                    f.write(f'{elemnt} ')
                    inter_bool = [objt.e==elemnt for objt in lattice.inter_atoms]
                    inter_order=inter_order+list(lattice.inter_positions[inter_bool])
        f.write(f'\n')
        ratio = (np.array(lattice.ratio)/(sum(lattice.ratio)))*(lattice.total) # Each unit cell have two atoms in BCC.
        for value in ratio:
            f.write(f'{round(value)} ')
        if show_inter==True:
            inter_ratio = (np.array(lattice.inter_ratio)/(sum(lattice.inter_ratio)))*(lattice.inter_total) # Each unit cell have two atoms in BCC.
            for val in inter_ratio:
                f.write(f'{round(val)} ')
        f.write(f'\n')
        if cartesian==False:
            f.write('Direct\n')
            for line in order:
                f.write(f'{line[0]} {line[1]} {line[2]}\n')
            for inline in inter_order:
                f.write(f'{inline[0]} {inline[1]} {inline[2]}\n') 
        else:
            f.write('Cartesian\n')
            for line in order:
                f.write(f'{line[0]} {line[1]} {line[2]}\n')
            for inline in inter_order:
                f.write(f'{inline[0]} {inline[1]} {inline[2]}\n')
        print("POSCAR Save File --- %s seconds ---" % (time.time() - start_time))
        
def Config_saveFile (output_file,lattice:Config, show_inter=False): #does not quite work... Make sure to fix
    with open(output_file, 'w') as f:
        f.write(f'Number of particles = {lattice.nx*lattice.ny*lattice.nz*len(lattice.unit_cell)}\n')
        f.write('A = 1.0 Angstrom (basic length-scale)\n')
        
        f.write(f'H0(1,1) = {lattice.basis_vectors[0][0]*lattice.nx} A\n')
        f.write(f'H0(1,2) = {lattice.basis_vectors[0][1]*lattice.nx} A\n')
        f.write(f'H0(1,3) = {lattice.basis_vectors[0][2]*lattice.nx} A\n')
        f.write(f'H0(2,1) = {lattice.basis_vectors[1][0]*lattice.ny} A\n')
        f.write(f'H0(2,2) = {lattice.basis_vectors[1][1]*lattice.ny} A\n')
        f.write(f'H0(2,3) = {lattice.basis_vectors[1][2]*lattice.ny} A\n')
        f.write(f'H0(3,1) = {lattice.basis_vectors[2][0]*lattice.nz} A\n')
        f.write(f'H0(3,2) = {lattice.basis_vectors[2][1]*lattice.nz} A\n')
        f.write(f'H0(3,3) = {lattice.basis_vectors[2][2]*lattice.nz} A\n')
        
        f.write(f'.NO_VELOCITY.\nentry_count = 3\n')

        if show_inter==True:
            count=0
            for line in lattice.frac_inter_positions:
                f.write(f'{lattice.inter_atoms[count].m}\n')
                f.write(f'{lattice.inter_atoms[count].e}\n')
                f.write(f'{line[0]} {line[1]} {line[2]}\n')
                count+=1
        count=0
        for line in lattice.frac_lat_positions:
            f.write(f'{lattice.atoms[count].m}\n')
            f.write(f'{lattice.atoms[count].e}\n')
            f.write(f'{line[0]} {line[1]} {line[2]}\n')
            count+=1

def kmc_main():
    hello = 5

start_time = time.time()
N = Element('N')
#print(N.e)
#print(N.m)

N_0 = Atom(N, 0)
#print(N_0.e)

bcc = Config('wurtzite',3,['Ga','N'], [1,1],['X','O','Si'],[100,2,1.5], 6,6,6, randm=True)
print(len(bcc.lat_positions))
print(len(bcc.nearest_neighbor[0]))
#print(bcc.lat_positions)
#for line in bcc.nearest_neighbor[3]:
#        print(len(line))
#print(fcc.nearest_neighbor)
POSCAR_saveFile ('../test.vasp',bcc, cartesian=True, show_inter=True)
print("Total --- %s seconds ---" % (time.time() - start_time))
