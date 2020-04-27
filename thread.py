import threading

def func1():
    for i in range(10):
        print("gibberish")

def func2():
    print("pass")

t1 = threading.Thread(target=func1)
t2 = threading.Thread(target=func2)


if __name__ == '__main__':
    t1.start()
    t2.start()