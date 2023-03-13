import CGRepresentation
import os
from pathlib import Path
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
        source_path = Path(__file__).resolve()
        source_dir = source_path.parent
        path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_nH_16S/16S_2/join_phylum_noduplicates/"
        counter = 1

        # Change the directory
        os.chdir(path)

        # Read text File

        def read_fasta_file(file_path):
            with open(file_path, 'r') as f:
                line = f.readline()
                self.sequence = line.replace("\n", "")


        # iterate through all file
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".fasta"):
                file_path = f"{path}\{file}"

                # call read text file function
                read_fasta_file(file_path)
                print(file_path)
                self.generate_dataset(counter)
                counter += 1

    def filter_sequence(self, sequence):

        filtered_sequence = MutableSeq(str(sequence))
        chars = ["Y", "N", "R", "M", "S", "W", "K", "D", "V", "B", "H", "P", "O"]

        if any(x in sequence for x in chars):

            char_count = self.count_char(sequence)

            for i in range(0, char_count):

                if filtered_sequence.find("Y") != -1:
                    filtered_sequence = filtered_sequence.replace("Y", "")

                elif filtered_sequence.find("N") != -1:
                    filtered_sequence = filtered_sequence.replace("N", "")

                elif filtered_sequence.find("R") != -1:
                    filtered_sequence = filtered_sequence.replace("R", "")

                elif filtered_sequence.find("M") != -1:
                    filtered_sequence = filtered_sequence.replace("M", "")

                elif filtered_sequence.find("S") != -1:
                    filtered_sequence = filtered_sequence.replace("S", "")

                elif filtered_sequence.find("W") != -1:
                    filtered_sequence = filtered_sequence.replace("W", "")

                elif filtered_sequence.find("D") != -1:
                    filtered_sequence = filtered_sequence.replace("D", "")

                elif filtered_sequence.find("B") != -1:
                    filtered_sequence = filtered_sequence.replace("B", "")

                elif filtered_sequence.find("H") != -1:
                    filtered_sequence = filtered_sequence.replace("H", "")

                elif filtered_sequence.find("V") != -1:
                    filtered_sequence = filtered_sequence.replace("V", "")

                elif filtered_sequence.find("K") != -1:
                    filtered_sequence = filtered_sequence.replace("K", "")

                elif filtered_sequence.find("P") != -1:
                    filtered_sequence = filtered_sequence.replace("P", "")

                elif filtered_sequence.find("O") != -1:
                    filtered_sequence = filtered_sequence.replace("O", "")

                else:
                    pass

        return filtered_sequence

    def count_char(self, sequence):
        tmp_sequence = MutableSeq(sequence)
        char_count = 0

        char_count += tmp_sequence.count("Y")
        char_count += tmp_sequence.count("N")
        char_count += tmp_sequence.count("R")
        char_count += tmp_sequence.count("M")
        char_count += tmp_sequence.count("S")
        char_count += tmp_sequence.count("W")
        char_count += tmp_sequence.count("K")
        char_count += tmp_sequence.count("D")
        char_count += tmp_sequence.count("H")
        char_count += tmp_sequence.count("V")
        char_count += tmp_sequence.count("B")
        char_count += tmp_sequence.count("P")
        char_count += tmp_sequence.count("O")

        return char_count

    def generate_dataset(self, counter):

        bio_sequence = self.filter_sequence(self.sequence)
        print("SEQUENZA UTILIZZATA: " + bio_sequence)
        print("COUNTER: " + str(counter))
        drawer = CGRepresentation.CGR(bio_sequence, self.CGR_type, self.outer_representation, self.rna_2structure)
        drawer.representation()
        drawer.plot(counter)


istanza_prova = CGRHandler("RNA", False, False)
CGRHandler.read_files(istanza_prova)