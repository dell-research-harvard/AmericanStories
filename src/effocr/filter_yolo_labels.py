import os
import shutil
import json

if __name__ == '__main__':
    os.chdir('C:\\Users\\bryan\\Downloads')
    num_found = 0
    unmatched_labels = []
    for label_name in os.listdir('labels_pablo\labels'):
        orig_label_name = '_'.join(label_name.split('.')[0].split('_')[:-1]) + '.txt'
        if os.path.isfile(os.path.join('labels_orig', orig_label_name)):
            with open(os.path.join('labels_orig', orig_label_name), 'r') as infile:
                orig_labels = [l.strip() for l in infile.readlines()]
            with open(os.path.join('labels_pablo\labels', label_name), 'r') as infile:
                new_labels = [l.strip() for l in infile.readlines()]
            any_unmatched = False
            for label in new_labels:
                c, x, y, w, h = map(float, label.split())
                matched = False
                for orig_label in orig_labels:
                    co, xo, yo, wo, ho = map(float, orig_label.split())
                    if c != co:
                        if abs(x - xo) < 0.001 and abs(y - yo) < 0.001 and abs(w - wo) < 0.001 and abs(h - ho) < 0.001:
                            matched = True
                            break
                if not matched:
                    any_unmatched = True
                    break
            if any_unmatched:
                unmatched_labels.append(label_name)
    
    for label in unmatched_labels:
        with open(os.path.join('labels_pablo', 'labels', label), 'r') as infile:
            pablo_labels = [l.strip() for l in infile.readlines()]
        for i in range(len(pablo_labels)):
            if pablo_labels[i][0] == '0':
                pablo_labels[i] = '1' + pablo_labels[i][1:]
            elif pablo_labels[i][0] == '1':
                pablo_labels[i] = '0' + pablo_labels[i][1:]

        with open(os.path.join('labels_pablo', 'labeled_labels', label), 'w') as outfile:
            outfile.write('\n'.join(pablo_labels))

    for label in unmatched_labels:
        img_name = label.split('.')[0] + '.jpg'
        shutil.copy(os.path.join('labels_pablo', 'images', img_name), os.path.join('labels_pablo', 'labeled_images', img_name))

    textlines = {label: '' for label in sorted(unmatched_labels)}
    with open('pablo_textlines.json', 'w') as outfile:
        json.dump(textlines, outfile, indent=4)