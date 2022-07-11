import pandas as pd
import numpy as np
if __name__ == "__main__":
    l1 = [["name1",[[1,0,1],[1,0,6]]],["name2",[[1,0,18],[1,0,8]]]]

    df1 = pd.DataFrame(l1)
    df1.to_csv("./datafram.csv",index=False)
    df2 = pd.read_csv("./datafram.csv")
    print(df2)

    l1 = ["name3",np.array([[1,0,18],[1,0,8]])]

    df1 = pd.DataFrame(l1)
    df1.to_csv("./datafram.csv",mode="a",index=False)
    df2 = pd.read_csv("./datafram.csv")
    print(df2)

    li = [[1,2,3],[4,5,6]]
    print(type(li[0]))
    a = np.array(li)
    print(type(a[0]))