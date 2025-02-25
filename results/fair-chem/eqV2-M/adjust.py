import os
import pickle

files = os.listdir('pt1')

for file in files:
    flattened_values = []
    output_file = f'pt1_{file}'
    with open(f'pt1/{file}', 'rb') as f:
        while True:
            try:
                data = pickle.load(f)  # Carrega um objeto do arquivo
            
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, list):
                            flattened_values.extend(item)  # Adiciona elementos individuais
                        else:
                            flattened_values.append(item)
                else:
                    flattened_values.append(data)

            except EOFError:
                break  # Final do arquivo

    with open(output_file, 'wb') as f:
        pickle.dump(flattened_values, f)

