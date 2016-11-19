#!/usr/bin/env python
#coding=utf-8

import smtplib
from email.mime.text import MIMEText

# 第三方 SMTP 服务
    # 邮件服务器地址
SMTPserver = "smtp.163.com"
    # 邮件用户名
username = "pclover11@163.com"
    # 密码
password = "1128zpc@"


# smtp会话过程中的mail from地址
from_addr = "pclover11@163.com"
# smtp会话过程中的rcpt to地址
to_addr = "zhaopengcheng11@aliyun.com"

# 信件内容
msg = MIMEText("this is another email",'plain',"utf-8")
msg['form'] = from_addr
msg['to'] =  to_addr
msg['subject'] = "just a test"

server = smtplib.SMTP(SMTPserver, 25)
server.login(username,password)
server.sendmail(from_addr, to_addr, msg.as_string())
server.quit()

print "send success"


