import numpy as np
import random
import time
import matplotlib.pyplot as plt

class Element:
    elements = ['X','Ga','N','H','Si','O', 'Mg']
    mass = [0,69.723,14.0067,1.00784,28.0855,15.999,24.305]

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
        Takes the unit cell and creates a larger structure based on the nx, ny, and nz inputs

        Parameters
        ---------------------------
        self:
        unit_cell: (ndarray) Positions of the unit cell
        nx: (int) x direction number of repetitions of unit cell
        ny: (int) y direction number of repetitions of unit cell
        nz: (int) z direction number of repetitions of unit cell
        '''
        start_time = time.time()
        direct_list = [[((i)+site[0])/nx, ((j)+site[1])/ny, ((k)+site[2])/nz] for k in range(nz) 
                       for j in range(ny) for i in range(nx) for site in unit_cell]

        print("Generate Supercell--- %s seconds ---" % (time.time() - start_time))
        return np.array(direct_list)
    
    def n_nearest_neighbor(self, positions, neighbor_dist:list):
        '''
        Outputs a list of the nearest neighbors for the structure.
        Format of list is: [nth nearest neighbor[n position[corresponding nearest neighbor positions]]]

        Parameters
        ---------------------------
        self:
        positions: (ndarray) All the positions in the structure
        maxNN_natoms:
        neighbor_dist: (list) The distance corresponding to each nearest neighbor, sorted from smallest to longest
        '''
        start_time = time.time()

        n_neighbor = []
        first_neigh = []
        second_neigh = []
        third_neigh = []
        fourth_neigh = []
        fifth_neigh = []
        sixth_neigh = []
        seventh_neigh = []
        eighth_neigh = []
        for coor in positions:
            dist = np.linalg.norm(positions-coor,axis=1) 
            first_bool = np.around(dist,5) == round(neighbor_dist[0],5)
            second_bool = np.around(dist,5) == round(neighbor_dist[1],5)
            third_bool = np.around(dist,5) == round(neighbor_dist[2],5)
            fourth_bool = np.around(dist,5) == round(neighbor_dist[3],5)
            fifth_bool = np.around(dist,5) == round(neighbor_dist[4],5)
            sixth_bool = np.around(dist,5) == round(neighbor_dist[5],5)
            seventh_bool = np.around(dist,5) == round(neighbor_dist[6],5)
            eighth_bool = np.around(dist,5) == round(neighbor_dist[7],5)
            first_neigh.append(list(positions[first_bool]) )
            second_neigh.append(list(positions[second_bool]) )
            third_neigh.append(list(positions[third_bool]) )
            fourth_neigh.append(list(positions[fourth_bool]) )
            fifth_neigh.append(list(positions[fifth_bool]) )
            sixth_neigh.append(list(positions[sixth_bool]) )
            seventh_neigh.append(list(positions[seventh_bool]) )
            eighth_neigh.append(list(positions[eighth_bool]) )
        
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
        n_neighbor.append(fifth_neigh)
        n_neighbor.append(sixth_neigh)
        n_neighbor.append(seventh_neigh)
        n_neighbor.append(eighth_neigh)

        print("Nearest Neighbor--- %s seconds ---" % (time.time() - start_time))
        return n_neighbor
    
    def Total_possible_events (self, positions, maxNN_natoms:int, neighbor_dist:list):
        '''
        
        '''
        n_neighbor = []
        first_neigh = []
        #second_neigh = []
        for coor in positions:
            dist = np.linalg.norm(positions-coor,axis=1)
            first_bool = np.around(dist,5) == round(neighbor_dist[0],5)
            #second_bool = np.around(dist,5) == round(neighbor_dist[1],5)
            first_neigh.append(list(positions[first_bool]) + list(-1*np.ones((maxNN_natoms-len(positions[first_bool]),3))))
            #second_neigh.append(list(positions[second_bool]) + list(-1*np.ones((maxNN_natoms-len(positions[second_bool]),3))))
        n_neighbor.append(first_neigh)
        #n_neighbor.append(second_neigh)

        total_possible_events_first = np.concatenate((np.repeat(positions,maxNN_natoms,axis=0),
                                                               np.reshape(np.array(first_neigh),(len(positions)*maxNN_natoms,3))),axis=1)
        #total_possible_events_second = np.concatenate((np.repeat(positions,maxNN_natoms,axis=0),
        #                                                        np.reshape(np.array(second_neigh),(len(positions)*maxNN_natoms,3))),axis=1)
        return n_neighbor,total_possible_events_first#,total_possible_events_second
            
    def cartesian_coor(self, positions, a,b,c, alpha, beta, gamma,nx:int,ny:int,nz:int):
        '''
        Converts fractional coordinates into cartesian coordinates

        Parameters
        ---------------------------
        self:
        positions: (ndarray) Positions you wish to convert into cartesian
        a: (float) length of lattice vector a
        b: (float) length of lattice vector b
        c: (float) length of lattice vector c
        alpha: (float) angle between b and c in degrees
        beta: (float) angle between a and c in degrees
        gamma: (float) angle between a and b in degrees
        nx: (int) x direction number of repetitions of unit cell
        ny: (int) y direction number of repetitions of unit cell
        nz: (int) z direction number of repetitions of unit cell
        '''
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

    def dopant_location(self, location, ):
        return #indexes for the atoms located in the proper area (list or ndarray)

    def __init__(self, struct:str, lattice_a:float, atom_types:list, ratio:list, dope_atoms:list, dope_numb:list, inter_atom_types:list, inter_ratio:list,
                  nx=1, ny=1, nz=1, randm=True, replace_first_atom=True, dope_location=['everywhere'], structures=structures):
        '''
        lattice_a is in Angstroms
        '''
        self.time = 0
        self.MSD = 0
        self.MSD_a = [0]
        self.MSD_b = [0]
        self.MSD_c = [0]
        self.all_occurred_events = []
        self.diffusivity = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self.struct = struct
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.a = lattice_a
        self.atom_types = atom_types
        self.ratio = ratio
        self.inter_atom_types = inter_atom_types
        self.inter_ratio = inter_ratio
        self.possible_dope_locations = ['everywhere','top-half','bottom-half']

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
            for i,a in enumerate(atom_list):
                if a.e == 'X':
                    atom_list[i] = 'X'
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                random.shuffle(atom_list)
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
            inter_atom_list = ['X' for atom in range(inter_total)]
            id_count=0
            for atom_index,amount in enumerate(inter_ratio):
                for additional in range(amount):
                    inter_num = random.randrange(0,len(inter_atom_list)-1)
                    inter_atom_list[inter_num]= Atom(Element(inter_atom_types[atom_index]),ID='INTER'+str(id_count))
                    id_count+=1
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                random.shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.maxNN_natoms = 12
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions, [(lattice_a*np.sqrt(3)/4),lattice_a*0.5, #Total nearest neighbors,
                                                                                 (lattice_a/np.sqrt(2)), np.sqrt(11)*0.25*lattice_a,       #includes both lattice and
                                                                                 np.sqrt(3)*0.5*lattice_a, lattice_a,                      #interstitial sites.
                                                                                 (np.sqrt(19)*lattice_a/4), np.sqrt(3/2)*lattice_a])
            
            #Total possible events for this system
            self.inter_nearest_neighbor,self.total_possible_events_first,self.total_possible_events_second = self.Total_possible_events(self.inter_positions, 
                                                                                                                                       self.maxNN_natoms,
                                                                                                                                       [(np.sqrt(3)*0.25*lattice_a),
                                                                                                                                        (lattice_a/np.sqrt(2))])
            #Making string versions of the positions and nearest neighbors
            self.str_inter_positions = np.array(['-1,-1,-1']+[",".join(item) for item in 
                                                            np.tile(self.inter_positions,(self.maxNN_natoms,1)).astype(str)])
            self.str_first_nearest_neighbor = np.array([",".join(item) for item in 
                                                        np.array(self.inter_nearest_neighbor[0]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])
            self.str_second_nearest_neighbor = np.array([",".join(item) for item in 
                                                         np.array(self.inter_nearest_neighbor[1]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])

            #finding the all_positions indices for the nearest neighbors ie self.all_positions[self.first_indices]==self.str_first_nearest_neighbor
            sorter = self.str_inter_positions.argsort(kind='mergesort')
            self.first_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_first_nearest_neighbor,sorter=sorter)]-1)
            self.second_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_second_nearest_neighbor,sorter=sorter)]-1)

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
            for i,a in enumerate(atom_list):
                if a.e == 'X':
                    atom_list[i] = 'X'
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                random.shuffle(atom_list)
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
            inter_atom_list = ['X' for atom in range(inter_total)]
            id_count=0
            for atom_index,amount in enumerate(inter_ratio):
                for additional in range(amount):
                    inter_num = random.randrange(0,len(inter_atom_list)-1)
                    inter_atom_list[inter_num]= Atom(Element(inter_atom_types[atom_index]),ID='INTER'+str(id_count))
                    id_count+=1
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                random.shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors  (Update Nearest Neighbors to include interstitial sites)
            self.maxNN_natoms = 24
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions, [0.5*lattice_a, np.sqrt(5)*0.25*lattice_a,
                                                                                 np.sqrt(2)*0.5*lattice_a,(np.sqrt(3)*lattice_a*0.5),
                                                                                 lattice_a, np.sqrt(13)*0.25*lattice_a,
                                                                                 (np.sqrt(2)*lattice_a),(np.sqrt(3)*lattice_a)])
            
            #Total possible events for this system
            self.inter_nearest_neighbor,self.total_possible_events_first,self.total_possible_events_second = self.Total_possible_events(self.inter_positions, 
                                                                                                                                       self.maxNN_natoms,
                                                                                                                                       [(lattice_a),#<--change these v
                                                                                                                                        (lattice_a)])#_______________j
            #Making string versions of the positions and nearest neighbors
            self.str_inter_positions = np.array(['-1,-1,-1']+[",".join(item) for item in 
                                                            np.tile(self.inter_positions,(self.maxNN_natoms,1)).astype(str)])
            self.str_first_nearest_neighbor = np.array([",".join(item) for item in 
                                                        np.array(self.inter_nearest_neighbor[0]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])
            self.str_second_nearest_neighbor = np.array([",".join(item) for item in 
                                                         np.array(self.inter_nearest_neighbor[1]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])

            #finding the all_positions indices for the nearest neighbors ie self.all_positions[self.first_indices]==self.str_first_nearest_neighbor
            sorter = self.str_inter_positions.argsort(kind='mergesort')
            self.first_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_first_nearest_neighbor,sorter=sorter)]-1)
            self.second_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_second_nearest_neighbor,sorter=sorter)]-1)

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
            for i,a in enumerate(atom_list):
                if a.e == 'X':
                    atom_list[i] = 'X'
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))
            if randm==True:
                random.shuffle(atom_list)
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
            inter_atom_list = ['X' for atom in range(inter_total)]
            id_count=0
            for atom_index,amount in enumerate(inter_ratio):
                for additional in range(amount):
                    inter_num = random.randrange(0,len(inter_atom_list)-1)
                    inter_atom_list[inter_num]= Atom(Element(inter_atom_types[atom_index]),ID='INTER'+str(id_count))
                    id_count+=1
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                random.shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.maxNN_natoms = 12
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions, [np.sqrt(3/8)*lattice_a, lattice_a/np.sqrt(2),
                                                                                 lattice_a, (5/(2*np.sqrt(6)))*lattice_a,
                                                                                 np.sqrt(11/8)*lattice_a, np.sqrt(3/2)*lattice_a,
                                                                                 np.sqrt(11/6)*lattice_a,np.sqrt(2)*lattice_a])

            #Total possible events for this system
            self.inter_nearest_neighbor,self.total_possible_events_first,self.total_possible_events_second = self.Total_possible_events(self.inter_positions, 
                                                                                                                                       self.maxNN_natoms,
                                                                                                                                       [(np.sqrt(3/8)*lattice_a),
                                                                                                                                        (lattice_a*np.sqrt(2/3))])
            #Making string versions of the positions and nearest neighbors
            self.str_inter_positions = np.array(['-1,-1,-1']+[",".join(item) for item in 
                                                            np.tile(self.inter_positions,(self.maxNN_natoms,1)).astype(str)])
            self.str_first_nearest_neighbor = np.array([",".join(item) for item in 
                                                        np.array(self.inter_nearest_neighbor[0]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])
            self.str_second_nearest_neighbor = np.array([",".join(item) for item in 
                                                         np.array(self.inter_nearest_neighbor[1]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])

            #finding the all_positions indices for the nearest neighbors ie self.all_positions[self.first_indices]==self.str_first_nearest_neighbor
            sorter = self.str_inter_positions.argsort(kind='mergesort')
            self.first_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_first_nearest_neighbor,sorter=sorter)]-1)
            self.second_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_second_nearest_neighbor,sorter=sorter)]-1)

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
            if len(element_types)>2 and randm==True:
                atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i in range(0,2) 
                            for position in range(round((ratio[i]/sum(ratio[0:2]))*total))]
                if replace_first_atom==True:
                    len_random = [0,len(fcc_lat_positions)-1]
                    self.ratio[0] = self.ratio[0] - sum(ratio[2:])
                else:
                    len_random = [len(fcc_lat_positions),len(atom_list)-1]
                    self.ratio[1] = self.ratio[1] - sum(ratio[2:])

                for inx,rat in enumerate(ratio[2:]):
                    for position in range(round((rat/sum(ratio))*total)):
                        in_num = random.randrange(len_random[0],len_random[1])
                        atom_list[in_num]=Atom(element_types[inx+2],'LAT_'+element_types[inx+2].e+str(position))

                if dope_atoms!=[]: 
                    dope_id_count=0
                    for dope_index,amount in enumerate(dope_numb):
                        for each_dope in range(amount):
                            atom_replace_num = random.randrange(len_random[0],len_random[1])
                            atom_list[atom_replace_num]= Atom(Element(dope_atoms[dope_index]),ID='LAT'+str(dope_id_count))
                            dope_id_count+=1

            else:
                atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i,r in enumerate(ratio) 
                            for position in range(round((r/sum(ratio))*total))]
                if dope_atoms!=[]:
                    if replace_first_atom==True:
                        len_random = [0,len(fcc_lat_positions)-1]
                    else:
                        len_random = [len(fcc_lat_positions),len(atom_list)-1]

                    dope_id_count=0
                    if randm==True:
                        for dope_index,amount in enumerate(dope_numb):
                            for each_dope in range(amount):
                                atom_replace_num = random.randrange(len_random[0],len_random[1])
                                atom_list[atom_replace_num]= Atom(Element(dope_atoms[dope_index]),ID='LAT'+str(dope_id_count))
                                dope_id_count+=1
                    else:
                        for dinx, dopant_amount in enumerate(dope_numb):
                            locations = self.dopant_location(dope_location[dinx])
                            for individual in range(dopant_amount):
                                atom_list[locations[individual]]=Atom(Element(dope_atoms[dinx]),ID='LAT'+str(dope_id_count))
                                dope_id_count+=1

            for i,a in enumerate(atom_list):
                if a.e == 'X':
                    atom_list[i] = 'X'
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))

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
            self.octa_positions = self.generate_supercell(self.octa_unit_cell, nx,ny,nz)
            self.tetra_positions = self.generate_supercell(self.tetra_unit_cell, nx,ny,nz)
            self.frac_inter_positions = np.array(list(self.octa_positions) + list(self.tetra_positions))
            self.inter_positions = self.cartesian_coor(self.frac_inter_positions,lattice_a,lattice_a,lattice_a,90,90,90,nx,ny,nz)

            #List of Atoms for Interstitial Sites
            inter_total = nx*ny*nz*len(self.inter_unit_cell)
            self.inter_total = inter_total
            inter_atom_list = ['X' for atom in range(inter_total)]
            id_count=0
            for atom_index,amount in enumerate(inter_ratio):
                for additional in range(amount):
                    if randm==True:
                        inter_num = random.randrange(0,len(inter_atom_list)-1)
                    else:
                        inter_num = id_count
                    inter_atom_list[inter_num]= Atom(Element(inter_atom_types[atom_index]),ID='INTER'+str(id_count))
                    id_count+=1
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                random.shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.maxNN_natoms=12
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions, [(lattice_a*np.sqrt(3)/4),lattice_a*0.5,
                                                                                 (lattice_a/np.sqrt(2)), np.sqrt(11)*0.25*lattice_a,
                                                                                 np.sqrt(3)*0.5*lattice_a, lattice_a,
                                                                                 (np.sqrt(19)*lattice_a/4), np.sqrt(3/2)*lattice_a])
            
            #Total possible events for this system
            self.nearest_dist = (np.sqrt(3)*0.25*lattice_a)
            self.inter_nearest_neighbor,self.total_possible_events_first = self.Total_possible_events(self.inter_positions, 
                                                                                                        self.maxNN_natoms,
                                                                                                        [self.nearest_dist]) #,self.total_possible_events_second
            #Making string versions of the positions and nearest neighbors
            self.str_inter_positions = np.array(['-1,-1,-1']+[",".join(item) for item in 
                                                            np.tile(self.inter_positions,(self.maxNN_natoms,1)).astype(str)])
            self.str_first_nearest_neighbor = np.array([",".join(item) for item in 
                                                        np.array(self.inter_nearest_neighbor[0]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])
            #self.str_second_nearest_neighbor = np.array([",".join(item) for item in 
            #                                             np.array(self.inter_nearest_neighbor[1]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])

            #finding the all_positions indices for the nearest neighbors ie self.all_positions[self.first_indices]==self.str_first_nearest_neighbor
            sorter = self.str_inter_positions.argsort(kind='mergesort')
            self.first_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_first_nearest_neighbor,sorter=sorter)]-1)
            self.true_octa = self.first_indices<(0.5*len(self.inter_positions))
            #self.second_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_second_nearest_neighbor,sorter=sorter)]-1)

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
            if len(element_types)>2 and randm==True:
                atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i in range(0,2) 
                            for position in range(round((ratio[i]/sum(ratio[0:2]))*total))]
                if replace_first_atom==True:
                    len_random = [0,len(hcp_lat_positions)-1]
                    self.ratio[0] = self.ratio[0] - sum(ratio[2:])
                else:
                    len_random = [len(hcp_lat_positions),len(atom_list)-1]
                    self.ratio[1] = self.ratio[1] - sum(ratio[2:])
                for inx,rat in enumerate(ratio[2:]):
                    for position in range(round((rat/sum(ratio))*total)):
                        in_num = random.randrange(len_random[0],len_random[1])
                        atom_list[in_num]=Atom(element_types[inx+2],'LAT_'+element_types[inx+2].e+str(position))
            else:
                atom_list = [Atom(element_types[i],'LAT_'+element_types[i].e+str(position)) for i,r in enumerate(ratio) 
                            for position in range(round((r/sum(ratio))*total))]
            for i,a in enumerate(atom_list):
                if a.e == 'X':
                    atom_list[i] = 'X'
            if len(atom_list)>len(self.lat_positions):
                atom_list.pop(-1)
            elif len(atom_list)<len(self.lat_positions):
                atom_list.append(Atom(element_types[-1],'LAT_'+element_types[-1].e+str(len(self.lat_positions))))

            self.atoms = atom_list

            #Interstitial Sites
            self.octa_unit_cell = [[1,0,0.25],
                                   [1,0,0.75]]

            self.tetra_unit_cell = [[(1/3),(2/3),(5/8)],
                                    [(2/3),(1/3),(1/8)]]
            self.inter_unit_cell = self.octa_unit_cell + self.tetra_unit_cell
            self.octa_positions = self.generate_supercell(self.octa_unit_cell, nx,ny,nz)
            self.tetra_positions = self.generate_supercell(self.tetra_unit_cell, nx,ny,nz)
            self.frac_inter_positions = np.array(list(self.octa_positions) + list(self.tetra_positions))
            self.inter_positions=self.cartesian_coor(self.frac_inter_positions,lattice_a,lattice_a,np.sqrt(8/3)*lattice_a,90,90,120,nx,ny,nz)

            #List of Atoms for Interstitial Sites
            inter_total = nx*ny*nz*len(self.inter_unit_cell)
            self.inter_total = inter_total
            inter_atom_list = ['X' for atom in range(inter_total)]
            id_count=0
            for atom_index,amount in enumerate(inter_ratio):
                for additional in range(amount):
                    if randm==True:
                        inter_num = random.randrange(0,len(inter_atom_list)-1)
                    else:
                        inter_num = id_count
                    inter_atom_list[inter_num]= Atom(Element(inter_atom_types[atom_index]),ID='INTER'+str(id_count))
                    id_count+=1
            if len(inter_atom_list)>len(self.inter_positions):
                inter_atom_list.pop(-1)
            elif len(inter_atom_list)<len(self.inter_positions):
                inter_atom_list.append(Atom(inter_element_types[-1],'INT_'+inter_element_types[-1].e+str(len(self.inter_positions))))
            if randm==True:
                random.shuffle(inter_atom_list)
                self.inter_atoms=inter_atom_list
            else:
                self.inter_atoms = inter_atom_list

            #Combining Lattice positions/atoms with Interstitial positions/atoms
            self.all_frac_positions = np.array(list(self.frac_lat_positions)+list(self.frac_inter_positions))
            self.all_positions = np.array(list(self.lat_positions)+list(self.inter_positions))
            self.all_atoms = self.atoms+self.inter_atoms

            #Nearest Neighbors
            self.maxNN_natoms=12
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions, [np.sqrt(3/8)*lattice_a, lattice_a/np.sqrt(2),
                                                                                 lattice_a, (5/(2*np.sqrt(6)))*lattice_a,
                                                                                 np.sqrt(11/8)*lattice_a, np.sqrt(3/2)*lattice_a,
                                                                                 np.sqrt(11/6)*lattice_a,np.sqrt(2)*lattice_a])
                                                                                 #np.sqrt(2)*lattice_a,np.sqrt(8/3)*lattice_a])

            #Total possible events for this system
            self.nearest_dist = (np.sqrt(3/8)*lattice_a)
            self.inter_nearest_neighbor,self.total_possible_events_first = self.Total_possible_events(self.inter_positions, 
                                                                                                                                       self.maxNN_natoms,
                                                                                                                                       [(np.sqrt(3/8)*lattice_a),
                                                                                                                                        (lattice_a*np.sqrt(2/3))]) #,self.total_possible_events_second
            #Making string versions of the positions and nearest neighbors
            self.str_inter_positions = np.array(['-1,-1,-1']+[",".join(item) for item in 
                                                            np.tile(self.inter_positions,(self.maxNN_natoms,1)).astype(str)])
            self.str_first_nearest_neighbor = np.array([",".join(item) for item in 
                                                        np.array(self.inter_nearest_neighbor[0]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])
            #self.str_second_nearest_neighbor = np.array([",".join(item) for item in 
            #                                             np.array(self.inter_nearest_neighbor[1]).reshape(len(self.inter_positions)*self.maxNN_natoms,3).astype(str)])

            #finding the all_positions indices for the nearest neighbors ie self.all_positions[self.first_indices]==self.str_first_nearest_neighbor
            sorter = self.str_inter_positions.argsort(kind='mergesort')
            self.first_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_first_nearest_neighbor,sorter=sorter)]-1)
            self.true_octa = self.first_indices<(0.5*len(self.inter_positions))
            #self.second_indices = np.array(sorter[np.searchsorted(self.str_inter_positions, self.str_second_nearest_neighbor,sorter=sorter)]-1)

        else: #-----------------------------------------------------------------------------------------------------------------
            print('ERROR:',struct,"is not an available structure:",structures)
            self.basis_vectors=np.array([[0,0,0],[0,0,0],[0,0,0]])

def POSCAR_saveFile (output_file,lattice:Config, cartesian=False, show_inter=False, show_X=True):
    '''
    Saves a POSCAR (.vasp) of the structure

    Parameters
    ---------------------------
    output_file: (str) Filename and location
    lattice: (Config) The structure you wish to create a .vasp file for
    cartesian: (bool) Whether or not the coordinates are cartesian or fractional. Default is cartesian=False
    show_inter: (bool) Whether or not to show the interstitial sites/atoms. Default is show_inter=False
    '''
    start_time = time.time()
    with open(output_file, 'w') as f:
        f.write(lattice.struct+' structure\n')
        f.write('1.0\n')
        f.write(f'{lattice.basis_vectors[0][0]*lattice.nx} {lattice.basis_vectors[0][1]*lattice.nx} {lattice.basis_vectors[0][2]*lattice.nx}\n')
        f.write(f'{lattice.basis_vectors[1][0]*lattice.ny} {lattice.basis_vectors[1][1]*lattice.ny} {lattice.basis_vectors[1][2]*lattice.ny}\n')
        f.write(f'{lattice.basis_vectors[2][0]*lattice.nz} {lattice.basis_vectors[2][1]*lattice.nz} {lattice.basis_vectors[2][2]*lattice.nz}\n')

        atomTypes,atomIndices,atomInverse,atomCounts = np.unique(np.array(lattice.atom_types+lattice.inter_atom_types), return_index=True, return_inverse=True, return_counts=True)
        poscar_atomNum = []
        order=[]
        x_in_atomtypes=False
        if cartesian==False:
            if show_inter==True:
                for elemnt in atomTypes:
                    if elemnt=='X':
                        if show_X==False:
                            continue
                        f.write(f'{elemnt} ')
                        xinter_bool = np.array(lattice.all_atoms)=='X'
                        order=order+list(lattice.all_frac_positions[xinter_bool])
                        x_in_atomtypes=True
                        poscar_atomNum.append(len(lattice.all_frac_positions[xinter_bool]))
                    else:
                        f.write(f'{elemnt} ')
                        notx = np.array(lattice.all_atoms)!= 'X'
                        notx_pos = lattice.all_frac_positions[notx]
                        inter_bool = [objt.e==elemnt for objt in np.array(lattice.all_atoms)[notx]]
                        order=order+list(notx_pos[inter_bool])
                        poscar_atomNum.append(len(notx_pos[inter_bool]))
                if x_in_atomtypes==False and show_X==True:
                    f.write('X')
                    xinter_bool = np.array(lattice.all_atoms)=='X'
                    order=order+list(lattice.all_frac_positions[xinter_bool])
                    poscar_atomNum.append(len(lattice.all_frac_positions[xinter_bool]))
            else:
                for elemnt in lattice.atom_types:
                    if elemnt=='X':
                        if show_X==False:
                            continue
                        f.write(f'{elemnt} ')
                        xinter_bool = np.array(lattice.atoms)=='X'
                        order=order+list(lattice.frac_lat_positions[xinter_bool])
                        poscar_atomNum.append(len(lattice.frac_lat_positions[xinter_bool]))
                    else:
                        f.write(f'{elemnt} ')
                        notx = np.array(lattice.atoms)!= 'X'
                        notx_pos = lattice.frac_lat_positions[notx]
                        inter_bool = [objt.e==elemnt for objt in np.array(lattice.atoms)[notx]]
                        order=order+list(notx_pos[inter_bool])
                        poscar_atomNum.append(len(notx_pos[inter_bool]))
        else:
            if show_inter==True:
                for elemnt in atomTypes:
                    if elemnt=='X':
                        if show_X==False:
                            continue
                        f.write(f'{elemnt} ')
                        xinter_bool = np.array(lattice.all_atoms)=='X'
                        order=order+list(lattice.all_positions[xinter_bool])
                        x_in_atomtypes=True
                        poscar_atomNum.append(len(lattice.all_positions[xinter_bool]))
                    else:
                        f.write(f'{elemnt} ')
                        notx = np.array(lattice.all_atoms)!= 'X'
                        notx_pos = lattice.all_positions[notx]
                        inter_bool = [objt.e==elemnt for objt in np.array(lattice.all_atoms)[notx]]
                        order=order+list(notx_pos[inter_bool])
                        poscar_atomNum.append(len(notx_pos[inter_bool]))
                if x_in_atomtypes==False and show_X==True:
                    f.write('X')
                    xinter_bool = np.array(lattice.all_atoms)=='X'
                    order=order+list(lattice.all_positions[xinter_bool])
                    poscar_atomNum.append(len(lattice.all_positions[xinter_bool]))
            else:
                for elemnt in lattice.atom_types:
                    if elemnt=='X':
                        if show_X==False:
                            continue
                        f.write(f'{elemnt} ')
                        xinter_bool = np.array(lattice.atoms)=='X'
                        order=order+list(lattice.lat_positions[xinter_bool])
                        poscar_atomNum.append(len(lattice.lat_positions[xinter_bool]))
                    else:
                        f.write(f'{elemnt} ')
                        notx = np.array(lattice.atoms)!= 'X'
                        notx_pos = lattice.lat_positions[notx]
                        inter_bool = [objt.e==elemnt for objt in np.array(lattice.atoms)[notx]]
                        order=order+list(notx_pos[inter_bool])
                        poscar_atomNum.append(len(notx_pos[inter_bool]))
        f.write('\n')
        for atomNum in poscar_atomNum:
            f.write(f'{atomNum} ')
        f.write('\n')
        if cartesian==False:
            f.write('Direct\n')
            for line in order:
                f.write(f'{line[0]} {line[1]} {line[2]}\n')
        else:
            f.write('Cartesian\n')
            for line in order:
                f.write(f'{line[0]} {line[1]} {line[2]}\n')
        print("POSCAR Save File --- %s seconds ---" % (time.time() - start_time))
        
def Config_saveFile (output_file,lattice:Config, show_inter=False): #does not quite work... Make sure to fix
    '''
    
    '''
    with open(output_file, 'w') as f:
        if show_inter==True:
            f.write(f'Number of particles = {lattice.total + lattice.inter_total}\n')
        else:
            f.write(f'Number of particles = {lattice.total}\n')
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

        count=0
        count_inter=0
        for line in lattice.frac_lat_positions:
            if lattice.atoms[count]=='X':
                f.write(f'0\n')
                f.write(f'{lattice.atoms[count]}\n')
                f.write(f'{line[0]} {line[1]} {line[2]}\n')
            else:
                f.write(f'{lattice.atoms[count].m}\n')
                f.write(f'{lattice.atoms[count].e}\n')
                f.write(f'{line[0]} {line[1]} {line[2]}\n')
            count+=1
        if show_inter==True:
            for line in lattice.frac_inter_positions:
                if lattice.inter_atoms[count_inter]=='X':
                    f.write(f'0\n')
                    f.write(f'{lattice.atoms[count_inter]}\n')
                    f.write(f'{line[0]} {line[1]} {line[2]}\n')
                else:
                    f.write(f'{lattice.inter_atoms[count_inter].m}\n')
                    f.write(f'{lattice.inter_atoms[count_inter].e}\n')
                    f.write(f'{line[0]} {line[1]} {line[2]}\n')
                count_inter+=1
        #print(count+count_inter)

def Magnitude(vector):
    return np.sqrt((vector[0]*vector[0]) + (vector[1]*vector[1]) + (vector[2]*vector[2]))

def Diffusivity_based_on_energy (E_a, delta_E_a, T:float, eV=True):
    '''
    
    '''
    D_0 = 10**(-6) #cm^2/s    change to a more accurate value
    k_B = 8.617333262 * (10**(-5)) #eV/K

    E_TtO = E_a + delta_E_a

    E_100 = 4*E_a + 2*delta_E_a
    E_010 = E_100
    E_001 = E_100

    E_110 = 2*E_a + delta_E_a
    E_101 = E_110
    E_011 = E_110

    if eV == False:
        conversion = 1.602 * (10**(-19)) #Coulombs
        E_100 = E_100/conversion
        E_010 = E_010/conversion
        E_001 = E_001/conversion
        E_110 = E_110/conversion
        E_101 = E_101/conversion
        E_011 = E_011/conversion

    #diffusion = np.array([[D_0*(np.exp(-E_100/(k_B*T))), D_0*(np.exp(-E_110/(k_B*T))), D_0*(np.exp(-E_101/(k_B*T)))],
    #                      [D_0*(np.exp(-E_110/(k_B*T))), D_0*(np.exp(-E_010/(k_B*T))), D_0*(np.exp(-E_100/(k_B*T)))],
    #                      [D_0*(np.exp(-E_101/(k_B*T))), D_0*(np.exp(-E_011/(k_B*T))), D_0*(np.exp(-E_001/(k_B*T)))]])

    diffusion = np.array([[(D_0*2*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T))), (D_0*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T))), (D_0*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T)))],
                          [(D_0*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T))), (D_0*2*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T))), (D_0*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T)))],
                          [(D_0*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T))), (D_0*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T))), (D_0*2*(np.exp(-E_a/(k_B*T)))+np.exp(-E_TtO/(k_B*T)))]])
    return diffusion

def MSD (crys:Config):
    events = np.array(crys.all_occurred_events)[:,:3]*10**(-8) #cm
    #MSD_a = []
    #MSD_b = []
    #MSD_c = []
    #for n in range(0,len(events)):
    #    MSD_a = MSD_a + [sum((events[n:,0]-events[n,0])**2)/(len(events)-n)]
    #    MSD_b = MSD_b + [sum((events[n:,1]-events[n,1])**2)/(len(events)-n)]
    #    MSD_c = MSD_c + [sum((events[n:,2]-events[n,2])**2)/(len(events)-n)]
    #MSD = np.array([[np.mean(MSD_a), 0, 0],
    #                [0, np.mean(MSD_b), 0],
    #                [0, 0, np.mean(MSD_c)]])
    A = crys.basis_vectors[0]
    B = crys.basis_vectors[1]
    C = crys.basis_vectors[2]
    A_tiled = np.tile(A, (len(events),1))
    B_tiled = np.tile(B, (len(events),1))
    C_tiled = np.tile(C, (len(events),1))

    events_a = (np.reshape(np.dot(events, A),(len(events),1))*A_tiled)/(np.dot(A,A))
    mag_events_a = np.sqrt(events_a[:,0]**2 + events_a[:,1]**2 + events_a[:,2]**2)
    events_b = (np.reshape(np.dot(events, B),(len(events),1))*B_tiled)/(np.dot(B,B))
    mag_events_b = np.sqrt(events_b[:,0]**2 + events_b[:,1]**2 + events_b[:,2]**2)
    events_c = (np.reshape(np.dot(events, C),(len(events),1))*C_tiled)/(np.dot(C,C))
    mag_events_c = np.sqrt(events_c[:,0]**2 + events_c[:,1]**2 + events_c[:,2]**2)

    #events_tiled_a = np.tril(np.tile(mag_events_a,(len(events),1))-np.reshape(mag_events_a,(len(events),1)),-1)
    #events_tiled_b = np.tril(np.tile(mag_events_b,(len(events),1))-np.reshape(mag_events_b,(len(events),1)),-1)
    #events_tiled_c = np.tril(np.tile(mag_events_c,(len(events),1))-np.reshape(mag_events_c,(len(events),1)),-1)

    len_N = -1*np.arange(len(events))+len(events)
    MSD_a_addition = ((-1*mag_events_a + mag_events_a[-1])**2)/len_N
    crys.MSD_a = list(np.array(crys.MSD_a)+MSD_a_addition)
    crys.MSD_a.append(0)

    MSD_b_addition = ((-1*mag_events_b + mag_events_b[-1])**2)/len_N
    crys.MSD_b = list(np.array(crys.MSD_b)+MSD_b_addition)
    crys.MSD_b.append(0)

    MSD_c_addition = ((-1*mag_events_c + mag_events_c[-1])**2)/len_N
    crys.MSD_c = list(np.array(crys.MSD_c)+MSD_c_addition)
    crys.MSD_c.append(0)

    #events_tiled_a = np.tril(np.tile(events[:,0],(len(events),1))-np.reshape(events[:,0],(len(events),1)),-1)
    #events_tiled_b = np.tril(np.tile(events[:,1],(len(events),1))-np.reshape(events[:,1],(len(events),1)),-1)
    #events_tiled_c = np.tril(np.tile(events[:,2],(len(events),1))-np.reshape(events[:,2],(len(events),1)),-1)

    #MSD_a = (sum((events_tiled_a**2))/len_N)[:-1]
    #MSD_b = (sum((events_tiled_b**2))/len_N)[:-1]
    #MSD_c = (sum((events_tiled_c**2))/len_N)[:-1]

    #MSD = np.array([[np.mean(MSD_a), 0, 0],
    #                [0, np.mean(MSD_b), 0],
    #                [0, 0, np.mean(MSD_c)]])
    
    return crys

def Found_diffusivity (crys:Config):
    a_vect = crys.basis_vectors[0]*(10**(-8)) #x in cm
    b_vect = crys.basis_vectors[1]*(10**(-8)) #y in cm
    c_vect = crys.basis_vectors[2]*(10**(-8)) #z in cm

    ab_vect = -1*a_vect + b_vect
    ba_vect = a_vect + b_vect
    bc_vect = -1*b_vect + c_vect
    cb_vect = b_vect + c_vect
    ac_vect = a_vect - c_vect
    ca_vect = a_vect + c_vect

    #planes for second-rank tensor
    V_aa = np.cross(b_vect,c_vect) #y
    V_ab = np.cross(ab_vect,c_vect) #y
    V_ac = np.cross(ac_vect,b_vect) #y

    V_ba = np.cross(c_vect, ba_vect) #m
    V_bb = np.cross(c_vect, a_vect) #y
    V_bc = np.cross(bc_vect,a_vect) #y

    V_ca = np.cross(ca_vect,b_vect) #m
    V_cb = np.cross(a_vect,cb_vect) #m
    V_cc = np.cross(a_vect,b_vect) #y

    all_events = np.array(crys.all_occurred_events)
    event_vectors = (all_events[:,3:] - all_events[:,:3])* (10**(-8)) #cm
    total_dist_traveled = sum( np.sqrt(event_vectors[:,0]**2 + event_vectors[:,1]**2 + event_vectors[:,2]**2) )  #cm
    #tot_vector = sum(abs(event_vectors)) #cm
    #tot_vector_dist = Magnitude(tot_vector)
    #total_x_comp = np.array([sum(event_vectors[:,0]), 0, 0])
    #total_y_comp = np.array([0, sum(event_vectors[:,1]), 0])
    #total_z_comp = np.array([0,0, sum(event_vectors[:,2])])

    V_aa_reshape = np.tile(V_aa, (len(event_vectors),1))
    V_ab_reshape = np.tile(V_ab, (len(event_vectors),1))
    V_ac_reshape = np.tile(V_ac, (len(event_vectors),1))

    V_ba_reshape = np.tile(V_ba, (len(event_vectors),1))
    V_bb_reshape = np.tile(V_bb, (len(event_vectors),1))
    V_bc_reshape = np.tile(V_bc, (len(event_vectors),1))

    V_ca_reshape = np.tile(V_ca, (len(event_vectors),1))
    V_cb_reshape = np.tile(V_cb, (len(event_vectors),1))
    V_cc_reshape = np.tile(V_cc, (len(event_vectors),1))

    per_aa_vect = (np.reshape(np.dot(event_vectors, V_aa),(len(event_vectors),1))*V_aa_reshape)/(np.dot(V_aa,V_aa))
    percent_aa = sum(( np.sqrt(per_aa_vect[:,0]**2 + per_aa_vect[:,1]**2 + per_aa_vect[:,2]**2) ))
    #percent_aa = sum(abs(per_aa_vect))
    #print(percent_aa/total_dist_traveled)
    per_ab_vect = (np.reshape(np.dot(event_vectors, V_ab),(len(event_vectors),1))*V_ab_reshape)/(np.dot(V_ab,V_ab))
    percent_ab = sum(( np.sqrt(per_ab_vect[:,0]**2 + per_ab_vect[:,1]**2 + per_ab_vect[:,2]**2) ))
    #percent_ab = sum(abs(np.dot(event_vectors, V_ab)))
    per_ac_vect = (np.reshape(np.dot(event_vectors, V_ac),(len(event_vectors),1))*V_ac_reshape)/(np.dot(V_ac,V_ac))
    percent_ac = sum(( np.sqrt(per_ac_vect[:,0]**2 + per_ac_vect[:,1]**2 + per_ac_vect[:,2]**2) ))
    #percent_ac = sum(abs(np.dot(event_vectors, V_ac)))

    per_ba_vect = (np.reshape(np.dot(event_vectors, V_ba),(len(event_vectors),1))*V_ba_reshape)/(np.dot(V_ba,V_ba))
    percent_ba = sum(( np.sqrt(per_ba_vect[:,0]**2 + per_ba_vect[:,1]**2 + per_ba_vect[:,2]**2) ))
    #percent_ba = sum(abs(np.dot(event_vectors, V_ba)))
    per_bb_vect = (np.reshape(np.dot(event_vectors, V_bb),(len(event_vectors),1))*V_bb_reshape)/(np.dot(V_bb,V_bb))
    percent_bb = sum(( np.sqrt(per_bb_vect[:,0]**2 + per_bb_vect[:,1]**2 + per_bb_vect[:,2]**2) ))
    #percent_bb = sum(abs(per_bb_vect))
    #percent_bb = sum(abs(np.dot(event_vectors, V_bb)))
    per_bc_vect = (np.reshape(np.dot(event_vectors, V_bc),(len(event_vectors),1))*V_bc_reshape)/(np.dot(V_bc,V_bc))
    percent_bc = sum(( np.sqrt(per_bc_vect[:,0]**2 + per_bc_vect[:,1]**2 + per_bc_vect[:,2]**2) ))
    #percent_bc = sum(abs(np.dot(event_vectors, V_bc)))

    per_ca_vect = (np.reshape(np.dot(event_vectors, V_ca),(len(event_vectors),1))*V_ca_reshape)/(np.dot(V_ca,V_ca))
    percent_ca = sum(( np.sqrt(per_ca_vect[:,0]**2 + per_ca_vect[:,1]**2 + per_ca_vect[:,2]**2) ))
    #percent_ca = sum(abs(np.dot(event_vectors, V_ca)))
    per_cb_vect = (np.reshape(np.dot(event_vectors, V_cb),(len(event_vectors),1))*V_cb_reshape)/(np.dot(V_cb,V_cb))
    percent_cb = sum(( np.sqrt(per_cb_vect[:,0]**2 + per_cb_vect[:,1]**2 + per_cb_vect[:,2]**2) ))
    #percent_cb = sum(abs(np.dot(event_vectors, V_cb)))
    per_cc_vect = (np.reshape(np.dot(event_vectors, V_cc),(len(event_vectors),1))*V_cc_reshape)/(np.dot(V_cc,V_cc))
    percent_cc = sum(( np.sqrt(per_cc_vect[:,0]**2 + per_cc_vect[:,1]**2 + per_cc_vect[:,2]**2) ))
    #percent_cc = sum(abs(per_cc_vect))
    #percent_cc = sum(abs(np.dot(event_vectors, V_cc)))

    #print(sum(abs(np.dot(event_vectors, V_ba)))/total_dist_traveled)

    diffusion_flux = ((len(all_events)*sum(crys.inter_ratio))/(crys.time*total_dist_traveled)) * np.array([[percent_aa/Magnitude(V_aa), percent_ab/Magnitude(V_ab), percent_ac/Magnitude(V_ac)],
                                                                                   [percent_ba/Magnitude(V_ba), percent_bb/Magnitude(V_bb), percent_bc/Magnitude(V_bc)],
                                                                                   [percent_ca/Magnitude(V_ca), percent_cb/Magnitude(V_cb), percent_cc/Magnitude(V_cc)]])
    
    #diffusion_flux = ((len(all_events)*sum(crys.inter_ratio))/(crys.time*total_dist_traveled)) * np.array([percent_aa/Magnitude(V_aa), 
    #                                                                                                       percent_bb/Magnitude(V_bb), 
    #                                                                                                       percent_cc/Magnitude(V_cc)])
    #print(diffusion_flux)
    A_aa = Magnitude(V_aa)
    #print(A_aa)
    #print((percent_aa+percent_ab+percent_ac+percent_ba+percent_bb+percent_bc+percent_ca+percent_cb+percent_cc)/total_dist_traveled)
    #print(((len(all_events)*sum(crys.inter_ratio))/(crys.time*total_dist_traveled))/A_aa)
    #print(percent_aa/Magnitude(V_aa))

    #gradient of the concentration --------------------------------------------------------------------------------------
    pos = crys.nx*0.5*crys.basis_vectors[0,0]*10**(-8)
    V_product = ( c_vect[0]*(a_vect[1]*b_vect[2]-a_vect[2]*b_vect[1]) 
                 - c_vect[1]*(a_vect[0]*b_vect[2]-a_vect[2]*b_vect[0]) 
                 + c_vect[2]*(a_vect[0]*b_vect[1]-a_vect[1]*b_vect[0]) )
    n_xyz = (np.array(crys.inter_atoms)!='X')
    x_coor, x_index = np.unique(crys.inter_positions[:,0]*10**(-8), return_index=True)
    id_care,x_inverse, x_counts = np.unique(crys.inter_positions[:,0]*n_xyz, return_inverse=True, return_counts=True)
    amount_x = n_xyz[x_index]*(x_counts[x_inverse])[x_index]
    N_x_derivative = np.poly1d(np.polyfit(x_coor,amount_x, 5)).deriv()
    #print(N_x_derivative(crys.nx*0.5*crys.basis_vectors[0,0]*10**(-8)))

    y_coor, y_index = np.unique(crys.inter_positions[:,1]*10**(-8), return_index=True)
    id_care,y_inverse, y_counts = np.unique(crys.inter_positions[:,1]*n_xyz, return_inverse=True, return_counts=True)
    amount_y = n_xyz[y_index]*(y_counts[y_inverse])[y_index]
    N_y_derivative = np.poly1d(np.polyfit(y_coor,amount_y, 5)).deriv()
    #print(N_y_derivative(crys.nx*0.5*crys.basis_vectors[0,0]*10**(-8)))

    z_coor, z_index = np.unique(crys.inter_positions[:,2]*10**(-8), return_index=True)
    id_care,z_inverse, z_counts = np.unique(crys.inter_positions[:,2]*n_xyz, return_inverse=True, return_counts=True)
    amount_z = n_xyz[z_index]*(z_counts[z_inverse])[z_index]
    N_z_derivative = np.poly1d(np.polyfit(z_coor,amount_z, 5)).deriv()
    #print(N_z_derivative(crys.nx*0.5*crys.basis_vectors[0,0]*10**(-8)))

    N_vect = np.array([N_x_derivative(pos),N_y_derivative(pos),N_z_derivative(pos)])
    N_xy = Magnitude(np.dot(N_vect, V_ab)*V_ab/(np.dot(V_ab,V_ab)))
    N_xz = Magnitude(np.dot(N_vect, V_ac)*V_ac/(np.dot(V_ac,V_ac)))

    N_yx = Magnitude(np.dot(N_vect, V_ba)*V_ba/(np.dot(V_ba,V_ba)))
    N_yz = Magnitude(np.dot(N_vect, V_bc)*V_bc/(np.dot(V_bc,V_bc)))
    
    N_zx = Magnitude(np.dot(N_vect, V_ca)*V_ca/(np.dot(V_ca,V_ca)))
    N_zy = Magnitude(np.dot(N_vect, V_cb)*V_cb/(np.dot(V_cb,V_cb)))

    gradient_concentration = ((-1*sum(crys.inter_ratio))/(crys.nx*crys.ny*crys.nz*V_product))*np.array([[abs(N_x_derivative(pos)), N_xy, N_xz],
                                                                                                        [N_yx, abs(N_y_derivative(pos)), N_yz],
                                                                                                        [N_zx, N_zy, abs(N_z_derivative(pos))]])

    diffusivity = -1*diffusion_flux/(gradient_concentration)

    return diffusivity

def Plane_corresponding_movement (crys:Config, possible_events:list):
    #doesn't work for wurtzite and probably hcp
    move_direction = np.array(possible_events)[:,:3] - np.array(possible_events)[:,3:]
    basis_vectors_tiled = np.tile(crys.basis_vectors, (len(possible_events),1,1))

    notyet_event_planes = np.round(np.linalg.solve(basis_vectors_tiled,move_direction),3)
    
    whole_num = np.reshape(np.repeat(np.min(notyet_event_planes, axis=1, initial=1, where=(notyet_event_planes!=0)), 3), (len(notyet_event_planes),3))
    event_planes = notyet_event_planes / whole_num

    return event_planes

def Corresponding_plane_energy (E_100:float, E_010:float, E_001:float, E_110:float, E_101:float, E_011:float, E_111:float, planes):
    plane_100 = (planes[:,0]==1)*(planes[:,1]==0)*(planes[:,2]==0) * E_100
    plane_010 = (planes[:,0]==0)*(planes[:,1]==1)*(planes[:,2]==0) * E_010
    plane_001 = (planes[:,0]==0)*(planes[:,1]==0)*(planes[:,2]==1) * E_001
    plane_110 = (planes[:,0]==1)*(planes[:,1]==1)*(planes[:,2]==0) * E_110
    plane_101 = (planes[:,0]==1)*(planes[:,1]==0)*(planes[:,2]==1) * E_101
    plane_011 = (planes[:,0]==0)*(planes[:,1]==1)*(planes[:,2]==1) * E_011
    plane_111 = (planes[:,0]==1)*(planes[:,1]==1)*(planes[:,2]==1) * E_111

    plane_energy = plane_100+plane_010+plane_001+plane_110+plane_101+plane_011+plane_111

    return np.reshape(plane_energy, (len(plane_energy),1))

def All_Events(crys:Config):
    '''
    Determines all possible events/pathways

    Parameters
    ---------------------------
    crys: (Config) Structure you are determining the events/pathways for
    '''
    #for indx, position in enumerate(crys.all_positions):
    #    if crys.all_atoms[indx].e != 'X':
    #        for final_one in crys.nearest_neighbor[0][indx]:
    #            atom_index_one = np.where((crys.all_positions[:,0]==final_one[0]) & (crys.all_positions[:,1]==final_one[1]) 
    #                                      & (crys.all_positions[:,2]==final_one[2]))[0][0]
    #            if crys.all_atoms[atom_index_one].e == 'X':
    #                Possible_Events.append([position, final_one])

    #        for final_two in crys.nearest_neighbor[1][indx]:
    #            atom_index_two = np.where((crys.all_positions[:,0]==final_two[0]) & (crys.all_positions[:,1]==final_two[1]) 
    #                                      & (crys.all_positions[:,2]==final_two[2]))[0][0]
    #            if crys.all_atoms[atom_index_two].e == 'X':
    #                Possible_Events.append([position, final_two])

    repeat_inter_atoms = np.repeat(crys.inter_atoms,crys.maxNN_natoms,axis=0)
    true_atom_bool = np.array(repeat_inter_atoms) != 'X' #(n*m)x1 bool of which are true atoms

    not_real_pos_first = (crys.total_possible_events_first[:,3]!=-1)*(crys.total_possible_events_first[:,4]!=-1)*(crys.total_possible_events_first[:,5]!=-1)
    #not_real_pos_second = (crys.total_possible_events_second[:,3]!=-1)*(crys.total_possible_events_second[:,4]!=-1)*(crys.total_possible_events_second[:,5]!=-1)
    available_position_first = (crys.first_indices != -1)*(np.array(crys.inter_atoms)[crys.first_indices]=='X')*not_real_pos_first
    #available_position_second = (crys.second_indices != -1)*(np.array(crys.inter_atoms)[crys.second_indices]=='X')*not_real_pos_second

    Possible_Events = (list(crys.total_possible_events_first[true_atom_bool*available_position_first]))#+
                       #list(crys.total_possible_events_second[true_atom_bool*available_position_second]))
    Octa_to_tetraBool = crys.true_octa[true_atom_bool*available_position_first]
            
    return Possible_Events, Octa_to_tetraBool

def rates_of_All_Events (crys:Config, possible_events:list, ifOcta__tetra:np.array,T, E_a, delta_E_a, q_inter, q_octa, q_tetra):
    '''
    
    '''
    #k_B = 1.380649*(10**(-23)) #Joules/Kelvin
    k_B = 8.617333262 * (10**(-5)) #eV/K

    #each of these should be an array of shape nx1 where n = len(possible_events)
    attempt_freq = (10**(13)) #s^-1

    #planes = abs(Plane_corresponding_movement(crys, possible_events))
    #energy_barrier = Corresponding_plane_energy(E_100, E_010, E_001, E_110, E_101, E_011, E_111, planes)
    e = 1.602 * 10**(-19) #C
    k_0 = 8.99 * 10**(9) #N*m^2/C^2
    r = crys.nearest_dist * 10**(-10) #m
    E_electric_octa = k_0 * (q_inter*q_octa*e)/r #
    E_electric_tetra = k_0 * (q_inter*q_tetra*e)/r
    delta_E_electric = E_electric_octa - E_electric_tetra

    octaToTetra = E_a + E_electric_octa #eV
    tetraToOcta = E_a + delta_E_a + delta_E_electric #eV
    energy_barrier = np.reshape(octaToTetra*ifOcta__tetra + tetraToOcta*(abs(ifOcta__tetra-1)),(len(possible_events),1))
    #energy_barrier = np.ones((len(possible_events),1)) * T_O_energy
    energy_initial = np.zeros((len(possible_events),1)) #change, found using vasp DFT?

    rates_AE = attempt_freq * np.exp(-1*(energy_barrier-energy_initial)/(k_B*T) )
    return rates_AE

def kMC_Main (crys:Config, diffusion, temp, E_a, delta_E_a, iteration, q_inter, q_octa, q_tetra):
    '''
    Determines an event/pathway and moves the atom accordingly

    Parameter
    ---------------------------
    crys: (Config) The structure you are altering
    '''
    possible_events, ifOctaToTetra=All_Events(crys)
    if possible_events==[]:
        return crys
    
    #Choose an event
    rates_of_events = rates_of_All_Events(crys, possible_events, ifOctaToTetra, temp, E_a, delta_E_a, q_inter, q_octa, q_tetra)
    W_k = np.cumsum(rates_of_events)
    random_num = random.randrange(0,1000)/1000
    event_occurs = np.where(W_k >= random_num*sum(rates_of_events))[0][0]
    crys.all_occurred_events.append(possible_events[event_occurs])

    #Atom moves
    initial_index = np.where((crys.all_positions[:,0]==possible_events[event_occurs][0]) 
                             & (crys.all_positions[:,1]==possible_events[event_occurs][1]) 
                             & (crys.all_positions[:,2]==possible_events[event_occurs][2]))[0][0]
    initial = crys.all_atoms[initial_index]
    final_index = np.where((crys.all_positions[:,0]==possible_events[event_occurs][3]) 
                             & (crys.all_positions[:,1]==possible_events[event_occurs][4]) 
                             & (crys.all_positions[:,2]==possible_events[event_occurs][5]))[0][0]
    final = crys.all_atoms[final_index]
    crys.all_atoms[initial_index] = final
    crys.all_atoms[final_index] = initial
    crys.atoms = crys.all_atoms[:len(crys.atoms)]
    crys.inter_atoms = crys.all_atoms[len(crys.atoms):]

    #Update Time
    time_random_num = random.randrange(1,1000)/1000
    delta_time = float(-1*(np.log(time_random_num))/(sum(rates_of_events)))
    crys.time = crys.time + delta_time

    #Diffusion = Found_diffusivity(crys)
    #if len(crys.all_occurred_events)==1:
    #    crys.MSD = np.array([[0,0,0],[0,0,0],[0,0,0]])
    #else:
    #    crys.MSD = MSD(crys)#2*3*Diffusion*(crys.time/iteration)*np.array([[4,2,2],[2,4,2],[2,2,4]]) #cm^2
    MSD(crys)
    crys.MSD = np.array([[np.mean(crys.MSD_a[:-1]), 0, 0],
                         [0, np.mean(crys.MSD_b[:-1]), 0],
                         [0, 0, np.mean(crys.MSD_c[:-1])]])
    crys.diffusivity = (crys.MSD*iteration)/(2*3*crys.time)
    #crys.diffusivity = Diffusion

    return crys

start_time = time.time()

bcc = Config('zincblende',4.27,['Ga','N'], [1,1], [], [], ['H'],[1], 1,1,1, randm=True)
print(len(bcc.all_positions))
print(len(bcc.nearest_neighbor[0]))
#print(bcc.lat_positions)
#for nn in bcc.nearest_neighbor:
#    nn_len=[]
#    for line in nn:
#        nn_len.append(len(line))
#    print(max(nn_len))
        #print(len(line))
#print(fcc.nearest_neighbor)

POSCAR_saveFile ('../test.vasp',bcc, cartesian=True, show_inter=True, show_X=False)
POSCAR_saveFile ('../test_sample_frac.vasp',bcc, cartesian=False, show_inter=True, show_X=False)
Config_saveFile ('../test.cfg',bcc, show_inter=True)
iteration_start_time = time.time()

temp = 300 #K
#e100 = 1 #eV
#e010 = 1
#e001 = 1
#e110 = 1
#e101 = 1
#e011 = 1
#e111 = 2

E_a = 0.2 #eV
delta_E_a = 0.3 #eV
q_inter = 0 #H
q_octa = -3 #N
q_tetra = 3 #Ga

diffusion = Diffusivity_based_on_energy(E_a, delta_E_a, T=temp) #cm^2/s
#print(diffusion)
MSD_x = []
MSD_y = []
MSD_z = []
kmc_time = []
for iteration in range(20000):
    kMC_Main(bcc, diffusion, temp, E_a, delta_E_a, iteration+1, q_inter, q_octa, q_tetra)
    MSD_x.append(bcc.MSD[0,0])
    MSD_y.append(bcc.MSD[1,1])
    MSD_z.append(bcc.MSD[2,2])
    kmc_time.append(bcc.time)
    #POSCAR_saveFile ('../'+str(iteration)+'_test.vasp',bcc, cartesian=True, show_inter=True, show_X=False)
    #print(crys_new.all_atoms==crys.all_atoms)
    #crys = crys_new
    if (iteration+1)%1000 ==0:
        print(iteration+1)
        print(bcc.diffusivity)
        #print(len(bcc.MSD_a))
        #print(np.mean(bcc.diffusivity))
        #print(bcc.MSD)
#print(bcc.time)
#print(bcc.MSD)
#print(bcc.diffusivity)
#print(bcc.MSD_a)
print("Iterations --- %s seconds ---" % (time.time() - iteration_start_time))
POSCAR_saveFile ('../final_test.vasp',bcc, cartesian=True, show_inter=True, show_X=False)
print("Total --- %s seconds ---" % (time.time() - start_time))

fig = plt.figure(figsize=(32,24))
plt.plot(kmc_time,MSD_x, color='green',label='X Direction', linewidth=8)
plt.plot(kmc_time,MSD_y, color='tab:blue',label='Y Direction', linewidth=8)
plt.plot(kmc_time,MSD_z, color='orange',label='Z Direction', linewidth=8)
plt.xlabel('Time (s)', size=60)
plt.xticks(fontsize=40)
plt.ylabel('Mean Squared Displacement (cm^2)', size=60)
plt.yticks(fontsize=50)
plt.title('Mean Squared Displacement vs Time', size=70)
plt.legend(fontsize=50)
plt.savefig('../poster/MSD_vs_time_1.png')
plt.show()
