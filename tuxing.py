import tkinter as tk  # 使用Tkinter前需要先导入
import os

# 实例化object，建立窗口window
window = tk.Tk()

# 给窗口的可视化起名字
window.title('谛听声纹识别系统')

# 设定窗口的大小(长 * 宽)
window.geometry('500x400')  # 这里的乘是小x

l = tk.Label(window, text='你好！欢迎使用本系统', bg='green', font=('Arial', 12), width=30, height=2)
# 说明： bg为背景，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l.pack()

# 第4步，在图形界面上设定标签
var = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
l = tk.Label(window, textvariable=var, bg='white', fg='black', font=('Arial', 12), width=30, height=1)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l.pack()

# 定义一个函数功能（内容自己自由编写），供点击Button按键时调用，调用命令参数command=函数名
on_hit = False
def hit_me():
    global on_hit
    if on_hit == False:
        on_hit = True
        var.set('加载中....')
        os.system("python predict_recognition.py")
    else:
        on_hit = False
        var.set('加载结束，请重试')


# 在窗口界面设置放置Button按键
b = tk.Button(window, text='进行录音检测', font=('Arial', 12), width=10, height=1, command=hit_me)
b.pack()

l = tk.Label(window, bg='red', fg='white', width=20, text='请劳烦您给这个程序打个分',height=-1)
l.pack()
#定义一个触发函数功能
def print_selection(value):
    l.config(text='您给这个项目打的分是 ' + value)
#创建一个尺度滑条，长度200字符，从0开始10结束，以2为刻度，精度为0.01，触发调用print_selection函数
s = tk.Scale(window, label='分数', from_=0, to=10, orient=tk.HORIZONTAL, length=200, showvalue=0,tickinterval=2, resolution=0.1, command=print_selection)
s.pack()

# 主窗口循环显示
window.mainloop()
