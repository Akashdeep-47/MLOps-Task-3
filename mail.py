
# Python code to illustrate Sending mail from
# your Gmail account
import smtplib

# creates SMTP session
s = smtplib.SMTP('smtp.gmail.com', 587)

# start TLS for security
s.starttls()

# Authentication
s.login("akashdeepg06@gmail.com", "guptadeep02")

# message to be sent
message = "Model was successfully trained with highest accuracy achieved"

# sending the mail
s.sendmail("akashdeepg06@gmail.com", "akashdeepg196@gmail.com", message)

# terminating the session
s.quit()
