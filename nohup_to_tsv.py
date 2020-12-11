try:
    f = open("nohup.out", "r")
    print("Finding trial text in nohup.out...")
    trials = []
    trial = {}
    current_trial = 0
    for x in f:
        if "Best Trial" in x:
            trials.append(trial)
            trial = {}
            break
        if "Trial " in x:
            current_trial = int(x[x.find('T') + 6: x.find('\\')])  # ['Trial 1\n']
            if current_trial != 1:
                trials.append(trial)
                trial = {}
        if current_trial != 0 and ("=" in x or "-" in x):
            if "=" in x:
                name = x[4: x.find('=') - 1]
                value = x[x.find('=') + 2: x.find('\\')]
            else:
                name = x[4: x.find('-') - 1]
                value = x[x.find('-') + 2: x.find('\\')]
            trial[name] = value
    f.close()

    data_keys = trials[0].keys()

    f2 = open("vectors.tsv", "w")
    print("Writing trial text to vectors.tsv...")

    # Print all values
    for i in range(len(trials)):
        trial = trials[i]
        trial_info = ""
        for key in data_keys:
            trial_info += trial[key] + "\t"
        if i == len(trials) - 1:
            f2.write(f"{trial_info[:-1]}")  # Don't write the last \t and the \n
        else:
            f2.write(f"{trial_info[:-1]}\n")  # Don't write the last \t

    f2.close()

    f3 = open("metadata.tsv", "w")
    print("Writing trial text to metadata.tsv...")
    # Print column labels
    column_labels = "Nr\t"
    for key in data_keys:
        column_labels += key + "\t"
    f3.write(f"{column_labels[:-1]}\n")  # Don't write the last \t

    # Print all values
    for i in range(len(trials)):
        trial = trials[i]
        trial_info = f"{i + 1}\t"
        # trial_info = ""
        for key in data_keys:
            trial_info += trial[key] + "\t"
        if i == len(trials) - 1:
            f3.write(f"{trial_info[:-1]}")  # Don't write the last \t and the \n
        else:
            f3.write(f"{trial_info[:-1]}\n")  # Don't write the last \t

    f3.close()

    print("Done! :)")

except FileNotFoundError:
    print("No nohup.out file was found!")
