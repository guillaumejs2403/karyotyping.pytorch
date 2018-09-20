
schedule_1 = [[125, 0.0001],
              [150, 0.00001],
              [175, 0.000001]]

schedule_2 = [[125, 0.0001],
              [175, 0.00001],
              [225, 0.000001]]

schedule_3 = [[125, 0.0001],
              [200, 0.00001],
              [300, 0.000001]]


def get_schedule(epochs_list,divider, base_lr):
    to_return = []
    for i in range(len(epochs_list)):
        if i == 0:
            app = [epochs_list[0],base_lr/divider]
        else:
            app = [epochs_list[i],app[1]/divider]
        to_return.append(app)
    return to_return
