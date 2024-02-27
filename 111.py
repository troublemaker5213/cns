import json

with open('111.txt','r',encoding='utf-8') as f:
    s=f.read()
    data=js = json.loads(json.dumps(eval(s)))
    list1 = sorted(data.items(), key=lambda x: x[1])

    for datas in list1:
        print(datas)
'''
36299c687c
b7fb871660
260dd9ad33
3f45a470ad
  258b3b33c6
  6a37a91708
'''