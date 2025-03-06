
"""
    def save_sample(self, output_path, sample_size=1000):
        #Speichert die zufällige Stichprobe in eine neue CSV-Datei.
        sample = self.get_sample(sample_size)
        if sample is not None:
            sample.to_csv(output_path, index=False)
            print(f"Stichprobe gespeichert in: {output_path}")
"""
# Teständerung
#if __name__ == "__main__":
#sampler.save_sample("sample_1000.csv") 
