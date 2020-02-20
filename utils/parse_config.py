def parse_cfg(cfgfile):
    file = open(cfgfile,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x)>0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    count = 0
    for line in lines:
        count  = count +1
        if line[0] == '[':
            if len(block) != 0 :
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        elif line.rstrip() == "Training" or line.rstrip() == "Testing":
            block["Mode"] = line.rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    print("parse cfg file success")
    return blocks   