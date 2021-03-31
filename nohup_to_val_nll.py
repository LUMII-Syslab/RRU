# This file was once used to retrieve the validation losses from nohup files, when they weren't logging correctly

import tensorflow as tf

try:
    f = open("nohup.out", "r")

    current_trial = 0
    current_epoch = 0
    validation_ago = -1
    for x in f:
        if "Starting training..." in x:
            current_trial += 1
            current_epoch = 0
            validation_ago = -1
            validation_writer = tf.summary.FileWriter(f"newlog/{current_trial}/validation")
        elif current_trial == 0:
            continue

        if "Final validation stats" in x:
            validation_ago = 0
        elif validation_ago >= 0:
            validation_ago += 1

        if validation_ago == 4:
            value = float(x[14: x.find(',')])
            loss_summary = tf.Summary()
            loss_summary.value.add(tag=f'validation_epoch_nll', simple_value=value)
            validation_writer.add_summary(loss_summary, current_epoch)
            validation_writer.flush()
            validation_ago = -1
            current_epoch += 1
    f.close()

    print("Done! :)")

except FileNotFoundError:
    print("No nohup.out file was found!")
