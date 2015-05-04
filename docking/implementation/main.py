import lennardjonespotential
import buildMolecule

def load_atoms(pdb_file):
    f = open(pdb_file,'r')
    pdb = f.readlines()
    f.close()

    lastresseqnr = 0

    # loop through every atom record
    for l in pdb:
        if l[0:6] == "ATOM  ":
            # TODO:
            # check to see if this is a second conformation of the previous atom
            # if len(atoms) != 0:
            #    if atoms[-1][17:26] == l[17:26]:
            #        continue

            atomname = l[12:15] # atom name               e.g. CG1
            resseqnr = l[22:26] # residue sequence number e.g. 3
            x        = l[30:37]
            y        = l[38:45]
            z        = l[46:53]

            if not resseqnr == lastresseqnr:
                # if we are at a new residue, start by adding this N as a child
                # to the last O
            lastresseqnr = resseqnr

            if atomname in [" N  "," CA "," C  "," O  "]:
                # backbone
            else:
                # not backbone, add as child to CA or last atom

            # TODO:
            # do stuff with this info and build a Node list
