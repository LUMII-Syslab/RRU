try:
    f = open("nohup.out", "r")
    worst_nll = None
    average_nll = 0.
    best_nll = None
    current_trial = 0.
    nan_counter = 0
    for x in f:
        if "Best Trial" in x:
            break
        if "Trial " in x:
            current_trial = int(x[x.find('T') + 6: x.find('\\')])  # ['Trial 1\n']
        if current_trial != 0 and "NLL" in x:
            if "nan" in x:
                nan_counter += 1
                continue
            else:
                value = float(x[x.find('-') + 2: x.find('\\')])
                average_nll += value
                if worst_nll is None or value > worst_nll[0]:
                    worst_nll = [value, current_trial]
                if best_nll is None or value < best_nll[0]:
                    best_nll = [value, current_trial]

    print(f"Worst NLL - {worst_nll[0]} @ Trial {worst_nll[1]}")
    print(f"Average NLL - {average_nll / current_trial}")
    print(f"Best NLL - {best_nll[0]} @ Trial {best_nll[1]}")
    print(f"NAN's encountered - {nan_counter}")
    f.close()


except FileNotFoundError:
    print("No nohup.out file was found!")
