l = ['A', 'B', 'C', 'D']
l2 = ['a', 'b', 'c', 'd']

for index, e in enumerate([enumerate(l), enumerate(l2)]):
    for idx, element in e:
        print(idx, element)
