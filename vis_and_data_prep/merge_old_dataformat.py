## Script Parameters ##
time_file = open("time.txt")  # txt file with time data in the old format
data_file = open("results_f1.txt")  # txt file with benchmark results in the old format
merged_file = open(
    "results_f1_new.txt", "a"
)  # txt file in which new format should be written

## DO NOT CHANGE BELOW ##
time_content = time_file.readlines()
data_content = data_file.readlines()
dataset_dict = {}
downstream_model = None
for i in range(0, len(data_content), 7):
    model_name = data_content[i].split(" ")[1].removesuffix("\n")
    downstream_name = data_content[i + 1].split(" ")[1].removesuffix("\n")
    dataset_name = data_content[i + 2].split(" ")[1].removesuffix("\n")
    sample_size = float(data_content[i + 3].split(" ")[2].removesuffix("\n"))
    org_loss = float(data_content[i + 4].split(" ")[2].removesuffix("\n"))
    syn_loss = float(data_content[i + 5].split(" ")[2].removesuffix("\n"))

    if downstream_model is None:
        downstream_model = downstream_name
    else:
        if downstream_model != downstream_name:
            print("Multiple Downstream models detected.")
            raise NotImplementedError

    if dataset_name not in dataset_dict.keys():
        dataset_dict[dataset_name] = {}

    if model_name not in dataset_dict[dataset_name].keys():
        dataset_dict[dataset_name][model_name] = {
            "size": [],
            "org": [],
            "syn": [],
            "time": [],
        }

    dataset_dict[dataset_name][model_name]["size"].append(sample_size)
    dataset_dict[dataset_name][model_name]["org"].append(org_loss)
    dataset_dict[dataset_name][model_name]["syn"].append(syn_loss)

for i in range(0, len(time_content), 5):
    model_name = time_content[i].split(" ")[1].removesuffix("\n")
    dataset_name = time_content[i + 1].split(" ")[1].removesuffix("\n")
    sample_size = float(time_content[i + 2].split(" ")[2].removesuffix("\n"))

    if dataset_name in ["california_housing", "microsoft", "year"]:
        continue

    if model_name in ["great", "topo"]:
        continue

    time = float(time_content[i + 3].split(" ")[1].removesuffix("\n"))

    dataset_dict[dataset_name][model_name]["time"].append(time)

for dataset in dataset_dict:
    for model in dataset_dict[dataset]:
        for i in range(0, len(dataset_dict[dataset][model]["org"])):
            merged_file.write("Model: " + model + "\n")
            merged_file.write("Downstream: " + downstream_model + "\n")
            merged_file.write("Dataset: " + dataset + "\n")
            merged_file.write(
                "Sample size: " + str(dataset_dict[dataset][model]["size"][i]) + "\n"
            )
            merged_file.write(
                "Original f1: " + str(dataset_dict[dataset][model]["org"][i]) + "\n"
            )
            merged_file.write(
                "Synthetic f1: " + str(dataset_dict[dataset][model]["syn"][i]) + "\n"
            )
            merged_file.write(
                "Time: " + str(dataset_dict[dataset][model]["time"][i]) + "\n"
            )
            merged_file.write("-----------------\n")

merged_file.close()
time_file.close()
data_file.close()
