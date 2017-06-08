content =\
    [line.strip().split() for line in open("ut_align_no_tag.a")]

f = open("ut_align_no_tag_clean.a", "w")
for line in content:
    for entry in line:
        if entry.find('?') != -1:
            l, rs = entry.split('?')
            rs = rs.split(',')
            for r in rs:
                f.write(l + '?' + r + " ")
        else:
            l, rs = entry.split('-')
            rs = rs.split(',')
            for r in rs:
                f.write(l + '-' + r + " ")
    f.write("\n")
f.close()
