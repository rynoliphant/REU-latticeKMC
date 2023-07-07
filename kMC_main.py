import numpy as np
import random
import time

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
    
    def n_nearest_neighbor(self, positions, maxNN_natoms, neighbor_dist:list):
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
            dist = np.linalg.norm(positions-coor,axis=1) #np.array([coor]).T
            first_bool = np.around(dist,5) == round(neighbor_dist[0],5)
            second_bool = np.around(dist,5) == round(neighbor_dist[1],5)
            third_bool = np.around(dist,5) == round(neighbor_dist[2],5)
            fourth_bool = np.around(dist,5) == round(neighbor_dist[3],5)
            fifth_bool = np.around(dist,5) == round(neighbor_dist[4],5)
            sixth_bool = np.around(dist,5) == round(neighbor_dist[5],5)
            seventh_bool = np.around(dist,5) == round(neighbor_dist[6],5)
            eighth_bool = np.around(dist,5) == round(neighbor_dist[7],5)
            first_neigh.append(list(positions[first_bool]) )#+ list(-1*np.ones((maxNN_natoms-len(positions[first_bool]),3))))
            second_neigh.append(list(positions[second_bool]) )#+ list(-1*np.ones((maxNN_natoms-len(positions[second_bool]),3))))
            third_neigh.append(list(positions[third_bool]) )#+ list(-1*np.ones((24-len(positions[third_bool]),3))))
            fourth_neigh.append(list(positions[fourth_bool]) )#+ list(-1*np.ones((24-len(positions[fourth_bool]),3))))
            fifth_neigh.append(list(positions[fifth_bool]) )#+ list(-1*np.ones((24-len(positions[fifth_bool]),3))))
            sixth_neigh.append(list(positions[sixth_bool]) )#+ list(-1*np.ones((24-len(positions[sixth_bool]),3))))
            seventh_neigh.append(list(positions[seventh_bool]) )#+ list(-1*np.ones((24-len(positions[seventh_bool]),3))))
            eighth_neigh.append(list(positions[eighth_bool]) )#+ list(-1*np.ones((24-len(positions[eighth_bool]),3))))
        
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
        n_neighbor = []
        first_neigh = []
        second_neigh = []
        for coor in positions:
            dist = np.linalg.norm(positions-coor,axis=1)
            first_bool = np.around(dist,5) == round(neighbor_dist[0],5)
            second_bool = np.around(dist,5) == round(neighbor_dist[1],5)
            first_neigh.append(list(positions[first_bool]) + list(-1*np.ones((maxNN_natoms-len(positions[first_bool]),3))))
            second_neigh.append(list(positions[second_bool]) + list(-1*np.ones((maxNN_natoms-len(positions[second_bool]),3))))
        n_neighbor.append(first_neigh)
        n_neighbor.append(second_neigh)

        total_possible_events_first = np.concatenate((np.repeat(positions,maxNN_natoms,axis=0),
                                                               np.reshape(np.array(first_neigh),(len(positions)*maxNN_natoms,3))),axis=1)
        total_possible_events_second = np.concatenate((np.repeat(positions,maxNN_natoms,axis=0),
                                                                np.reshape(np.array(second_neigh),(len(positions)*maxNN_natoms,3))),axis=1)
        return n_neighbor,total_possible_events_first,total_possible_events_second
            
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

    def __init__(self, struct:str, lattice_a:float, atom_types:list, ratio:list, inter_atom_types:list, inter_ratio:list,
                  nx=1, ny=1, nz=1, randm=False, replace_first_atom=True, structures=structures):
        '''
        '''
        self.time = 0
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
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions,self.maxNN_natoms, [(lattice_a*np.sqrt(3)/4),lattice_a*0.5, #Total nearest neighbors,
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
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions, self.maxNN_natoms, [0.5*lattice_a, np.sqrt(5)*0.25*lattice_a,
                                                                                 np.sqrt(2)*0.5*lattice_a,(np.sqrt(3)*lattice_a*0.5),
                                                                                 lattice_a, np.sqrt(13)*0.25*lattice_a,
                                                                                 (np.sqrt(2)*lattice_a),(np.sqrt(3)*lattice_a)])
            
            self.total_possible_events_first = np.concatenate((np.repeat(self.all_positions,self.maxNN_natoms,axis=0),
                                                               np.reshape(np.array(self.nearest_neighbor[0]),(len(self.all_positions)*self.maxNN_natoms,3))),axis=1)
            self.total_possible_events_second = np.concatenate((np.repeat(self.all_positions,self.maxNN_natoms,axis=0),
                                                                np.reshape(np.array(self.nearest_neighbor[1]),(len(self.all_positions)*self.maxNN_natoms,3))),axis=1)

            #Making string versions of the positions and nearest neighbors
            self.str_all_positions = np.array(['-1,-1,-1']+[",".join(item) for item in 
                                                            np.tile(self.all_positions,(self.maxNN_natoms,1)).astype(str)])
            self.str_first_nearest_neighbor = np.array([",".join(item) for item in 
                                                        np.array(self.nearest_neighbor[0]).reshape(len(self.all_positions)*self.maxNN_natoms,3).astype(str)])
            self.str_second_nearest_neighbor = np.array([",".join(item) for item in 
                                                         np.array(self.nearest_neighbor[1]).reshape(len(self.all_positions)*self.maxNN_natoms,3).astype(str)])

            #finding the all_positions indices for the nearest neighbors ie self.all_positions[self.first_indices]==self.str_first_nearest_neighbor
            sorter = self.str_all_positions.argsort(kind='mergesort')
            self.first_indices = np.array(sorter[np.searchsorted(self.str_all_positions, self.str_first_nearest_neighbor,sorter=sorter)]-1)
            self.second_indices = np.array(sorter[np.searchsorted(self.str_all_positions, self.str_second_nearest_neighbor,sorter=sorter)]-1)

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
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions,self.maxNN_natoms, [np.sqrt(3/8)*lattice_a, lattice_a/np.sqrt(2),
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
            self.maxNN_natoms=12
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions,self.maxNN_natoms, [(lattice_a*np.sqrt(3)/4),lattice_a*0.5,
                                                                                 (lattice_a/np.sqrt(2)), np.sqrt(11)*0.25*lattice_a,
                                                                                 np.sqrt(3)*0.5*lattice_a, lattice_a,
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
            self.maxNN_natoms=12
            self.nearest_neighbor = self.n_nearest_neighbor(self.all_positions,self.maxNN_natoms, [np.sqrt(3/8)*lattice_a, lattice_a/np.sqrt(2),
                                                                                 lattice_a, (5/(2*np.sqrt(6)))*lattice_a,
                                                                                 np.sqrt(11/8)*lattice_a, np.sqrt(3/2)*lattice_a,
                                                                                 np.sqrt(11/6)*lattice_a,np.sqrt(2)*lattice_a])
                                                                                 #np.sqrt(2)*lattice_a,np.sqrt(8/3)*lattice_a])

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

        else: #-----------------------------------------------------------------------------------------------------------------
            print('ERROR:',struct,"is not an available structure:",structures)
            self.basis_vectors=np.array([[0,0,0],[0,0,0],[0,0,0]])

def POSCAR_saveFile (output_file,lattice:Config, cartesian=False, show_inter=False):
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
                    f.write(f'{elemnt} ')
                    if elemnt=='X':
                        xinter_bool = np.array(lattice.all_atoms)=='X'
                        order=order+list(lattice.all_frac_positions[xinter_bool])
                        x_in_atomtypes=True
                        poscar_atomNum.append(len(lattice.all_frac_positions[xinter_bool]))
                    else:
                        notx = np.array(lattice.all_atoms)!= 'X'
                        notx_pos = lattice.all_positions[notx]
                        inter_bool = [objt.e==elemnt for objt in np.array(lattice.all_atoms)[notx]]
                        order=order+list(notx_pos[inter_bool])
                        poscar_atomNum.append(len(notx_pos[inter_bool]))
                if x_in_atomtypes==False:
                    f.write('X')
                    xinter_bool = np.array(lattice.all_atoms)=='X'
                    order=order+list(lattice.all_frac_positions[xinter_bool])
                    poscar_atomNum.append(len(lattice.all_frac_positions[xinter_bool]))
            else:
                for elemnt in lattice.atom_types:
                    f.write(f'{elemnt} ')
                    if elemnt=='X':
                        xinter_bool = np.array(lattice.atoms)=='X'
                        order=order+list(lattice.frac_lat_positions[xinter_bool])
                        poscar_atomNum.append(len(lattice.frac_lat_positions[xinter_bool]))
                    else:
                        notx = np.array(lattice.atoms)!= 'X'
                        notx_pos = lattice.frac_lat_positions[notx]
                        inter_bool = [objt.e==elemnt for objt in np.array(lattice.atoms)[notx]]
                        order=order+list(notx_pos[inter_bool])
                        poscar_atomNum.append(len(notx_pos[inter_bool]))
        else:
            if show_inter==True:
                for elemnt in atomTypes:
                    f.write(f'{elemnt} ')
                    if elemnt=='X':
                        xinter_bool = np.array(lattice.all_atoms)=='X'
                        order=order+list(lattice.all_positions[xinter_bool])
                        x_in_atomtypes=True
                        poscar_atomNum.append(len(lattice.all_positions[xinter_bool]))
                    else:
                        notx = np.array(lattice.all_atoms)!= 'X'
                        notx_pos = lattice.all_positions[notx]
                        inter_bool = [objt.e==elemnt for objt in np.array(lattice.all_atoms)[notx]]
                        order=order+list(notx_pos[inter_bool])
                        poscar_atomNum.append(len(notx_pos[inter_bool]))
                if x_in_atomtypes==False:
                    f.write('X')
                    xinter_bool = np.array(lattice.all_atoms)=='X'
                    order=order+list(lattice.all_positions[xinter_bool])
                    poscar_atomNum.append(len(lattice.all_positions[xinter_bool]))
            else:
                for elemnt in lattice.atom_types:
                    f.write(f'{elemnt} ')
                    if elemnt=='X':
                        xinter_bool = np.array(lattice.atoms)=='X'
                        order=order+list(lattice.lat_positions[xinter_bool])
                        poscar_atomNum.append(len(lattice.lat_positions[xinter_bool]))
                    else:
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
    not_real_pos_second = (crys.total_possible_events_second[:,3]!=-1)*(crys.total_possible_events_second[:,4]!=-1)*(crys.total_possible_events_second[:,5]!=-1)
    available_position_first = (crys.first_indices != -1)*(np.array(crys.inter_atoms)[crys.first_indices]=='X')*not_real_pos_first
    available_position_second = (crys.second_indices != -1)*(np.array(crys.inter_atoms)[crys.second_indices]=='X')*not_real_pos_second

    Possible_Events = (list(crys.total_possible_events_first[true_atom_bool*available_position_first])+
                       list(crys.total_possible_events_second[true_atom_bool*available_position_second]))
            
    return Possible_Events

def kMC_Main (crys:Config):
    '''
    Determines an event/pathway and moves the atom accordingly

    Parameter
    ---------------------------
    crys: (Config) The structure you are altering
    '''
    possible_events=All_Events(crys)
    random_num = random.randrange(0,100)/100
    if possible_events==[]:
        return crys

    #will defiantely change later to choose an event more in align with its probability of happening
    event_occurs = random.randrange(0,len(possible_events)-1)
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
    rates_of_events = np.ones((len(possible_events))) #will change later to accurately represent rates of events
    time_random_num = random.randrange(1,100)/100
    delta_time = -1*(np.log(time_random_num))/(sum(rates_of_events))
    crys.time = crys.time + delta_time

    #print(initial==crys.all_atoms[initial_index])

    return crys

start_time = time.time()
N = Element('N')
#print(N.e)
#print(N.m)

N_0 = Atom(N, 0)
#print(N_0.e)

bcc = Config('zincblende',3,['Ga','N', 'Mg'], [1,1,0.1],['H','Si'],[10,5], 3,3,3, randm=True)
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

POSCAR_saveFile ('../test.vasp',bcc, cartesian=True, show_inter=True)
iteration_start_time = time.time()
for iteration in range(500):
    kMC_Main(bcc)
    #print(crys_new.all_atoms==crys.all_atoms)
    #crys = crys_new
#print(bcc.time)
print("Iterations --- %s seconds ---" % (time.time() - iteration_start_time))
POSCAR_saveFile ('../final_test.vasp',bcc, cartesian=True, show_inter=True)
print("Total --- %s seconds ---" % (time.time() - start_time))
