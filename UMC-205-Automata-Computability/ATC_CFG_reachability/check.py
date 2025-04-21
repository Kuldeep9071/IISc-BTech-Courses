with open("res.txt", "r") as f1, open("out.txt", "r") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

if lines1 != lines2:
    print("The files are different.")

