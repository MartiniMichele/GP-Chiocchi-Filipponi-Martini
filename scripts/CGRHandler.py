import CGRepresentation
import os
from Bio import AlignIO
from Bio.Seq import MutableSeq


class CGRHandler:
    # in futuro cambiare il costruttore per poter scegliere le modalità della CGR
    def __init__(self, CGR_type, outer_representation, rna_2structure):
        self.CGR_type = CGR_type
        self.outer_representation = outer_representation
        self.rna_2structure = rna_2structure
        self.sequence = None


    def read_files(self):

        # Folder Path
        path = "C:/Users/Michele/Documents/GitHub/GP-Chiocchi-Filipponi-Martini/Fasta_5S"
        counter = 1

        # Change the directory
        os.chdir(path)

        # Read text File

        def read_text_file(file_path):
            with open(file_path, 'r') as f:
                alignment = AlignIO.read(f, 'fasta')
                seq = [record.seq for record in alignment]
                self.sequence = seq

        # iterate through all file
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".fasta"):
                file_path = f"{path}\{file}"

                # call read text file function
                read_text_file(file_path)
                self.generate_dataset(counter)
                counter += 1

    def filter_sequence(self, sequence):

        filtered_sequence = MutableSeq(str(sequence[0]))
        bio_sequence = MutableSeq(str(sequence[0]))

        if bio_sequence.find("Y") != -1:
            filtered_sequence = bio_sequence.replace("Y", "")

        elif bio_sequence.find("N") != -1:
            filtered_sequence = bio_sequence.replace("N", "")

        elif bio_sequence.find("R") != -1:
            filtered_sequence = bio_sequence.replace("R", "")

        elif bio_sequence.find("M") != -1:
            filtered_sequence = bio_sequence.replace("M", "")

        elif bio_sequence.find("S") != -1:
            filtered_sequence = bio_sequence.replace("S", "")

        else:
            pass

        return filtered_sequence

    def generate_dataset(self, counter):

        bio_sequence = self.filter_sequence(self.sequence)
        print("SEQUENZA UTILIZZATA: " + bio_sequence)
        print("COUNTER: " + str(counter))
        drawer = CGRepresentation.CGR(bio_sequence, self.CGR_type, self.outer_representation, self.rna_2structure)
        drawer.representation()
        drawer.plot(counter)


istanza_prova = CGRHandler("RNA", False, False)
CGRHandler.read_files(istanza_prova)
