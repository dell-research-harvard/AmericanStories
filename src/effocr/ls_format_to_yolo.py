import json

if __name__ == '__main__':
    with open(r'C:\Users\bryan\Downloads\shao-yu_labels.json', 'r') as infile:
        labels = json.load(infile)

    with open(r'C:\Users\bryan\Downloads\shao-yu_labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    label_map = {'char': 0,
                 'word': 1,}
    
    textlines = {}
    for img in labels:
        # verify the label has been corrected
        if img['annotations'][0]['completed_by'] != 1:
            continue

        filename = img['data']['img'].split('/')[-1]

        yolo_labels = []
        for annotation in img['annotations'][0]['result']:
            if 'labels' in annotation['value'].keys():
                c = annotation['value']['labels'][0]
                x = annotation['value']['x']
                y = annotation['value']['y']
                w = annotation['value']['width']
                h = annotation['value']['height']

                y0 = y + h / 2
                x0 = x + w / 2

                yolo_labels.append(f'{label_map[c]} {x0 / 100} {y0 / 100} {w / 100} {h / 100}')

            elif 'text' in annotation['value'].keys():
                textlines[filename] = annotation['value']['text'][0]

        if filename in textlines.keys():
            with open(f'C:\\Users\\bryan\\Downloads\\shao-yu_labels\\{filename.split(".")[0]}.txt', 'w') as outfile:
                outfile.write('\n'.join(yolo_labels))

    with open('C:\\Users\\bryan\\Downloads\\shao-yu_textlines.json', 'w') as outfile:
        json.dump(textlines, outfile, indent=4)
                
    print(len(textlines))
    print(len(labels))

