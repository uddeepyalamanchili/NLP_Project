def convert_to_srctrg(inp, src, trg):
    bad_words = ('### comment:', '### contributor:', '### id:', '### page:', '### timestamp:', '###')
    i = 0
    for line in inp:
        if line.isspace():
            continue
        if line.startswith(bad_words, 0):
            continue
        if i % 2 == 0:
            src.write(line)
        else:
            trg.write(line)
        i = i + 1


convert_to_srctrg(open('train'), open('data/train_merge.src','w'), open('data/train_merge.tgt','w'))
convert_to_srctrg(open('val'), open('data/valid.src','w'), open('data/valid.tgt','w'))