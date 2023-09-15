import os

# Constants
TEST_SIZE_THRESHOLD = 40

def parse_clusters(cluster_data):
    clusters = []
    lines = cluster_data.split("\n")
    cluster = {}
    for line in lines:
        if "Cluster Number:" in line:
            if cluster:
                clusters.append(cluster)
            cluster = {"cluster_info": line, "data": []}
        elif "Accuracy" in line or "Test set length" in line:
            cluster["data"].append(line)
    if cluster:
        clusters.append(cluster)
    return clusters

def is_cluster_above_threshold(cluster):
    for data in cluster["data"]:
        if "Accuracy" in data:
            accuracy_value = float(data.split(" ")[1])
            if accuracy_value > 70:
                return True
    return False

def is_test_size_above_threshold(cluster):
    for data in cluster["data"]:
        if "Test set length" in data:
            test_size_value = int(data.split(": ")[-1])
            if test_size_value > TEST_SIZE_THRESHOLD:
                return True
    return False

def extract_data(filename):
    with open(filename, "r") as f:
        content = f.read()
    header, cluster_data = content.split("\n\n", 1)
    return header, parse_clusters(cluster_data)

def main():
    root_dir = "."
    total_results_file = "totalResults.txt"

    with open(total_results_file, "w") as result_f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if dirpath.startswith("./2023"):
                if "results.txt" in filenames:
                    full_path = os.path.join(dirpath, "results.txt")
                    header, clusters = extract_data(full_path)

                    # Add file path to header and write it to the result file
                    header_with_path = f"Path: {full_path}\n{header}"
                    result_f.write(header_with_path)
                    result_f.write("\n\n")

                    # Write clusters with accuracy > 65% to the result file
                    for cluster in clusters:
                        if is_cluster_above_threshold(cluster) and is_test_size_above_threshold(cluster):
                            result_f.write(cluster["cluster_info"])
                            result_f.write("\n")
                            result_f.write("\n".join(cluster["data"]))
                            result_f.write("\n\n")

if __name__ == "__main__":
    main()
