import pandas as pd
import matplotlib.pyplot as plt

class RRUFF_text:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe:pd.DataFrame = self._read_file()

    def _normalize_list(self, input_list):
        # Find the minimum and maximum values in the list
        min_val = min(input_list)
        max_val = max(input_list)

        # Check if all values in the list are the same (avoid division by zero)
        if min_val == max_val:
            return [0.0] * len(input_list)  # Return a list of zeros

        # Normalize each value in the list to the range [0, 1]
        normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

        return normalized_list

    def _read_file(self):
        # Read the text file, skipping the header lines (lines starting with "##")
        with open(self.file_path, 'r') as file:
            data_lines = [line.strip() for line in file.readlines() if not line.startswith('##')]
        
        # Create a DataFrame from the data lines
        df = pd.DataFrame([line.split(',') for line in data_lines], columns=['wavenumbers', 'intensities'])
        df = df.dropna(subset=['intensities'])
        for col in df.columns:
            df[col] = [float(i) for i in df[col]]
        
        df['intensities_normalized'] = self._normalize_list(list(df['intensities']))


        return df

    def plot_columns(self, normalize = True):
        # Plot 'Column1' over 'Column2'
        plt.figure(figsize=(8, 6))
        if normalize:
            intensites = self.dataframe['intensities_normalized']
        else:
            intensites = self.dataframe['intensities']
        plt.plot(self.dataframe['wavenumbers'], intensites)
        plt.xlabel('wavenumbers')
        plt.ylabel('intensities')
        plt.title('wavenumbers vs. intensities')
        plt.show()
