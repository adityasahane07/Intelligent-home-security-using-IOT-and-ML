from gpiozero import Servo
from time import sleep

from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory()

servo = Servo(12, pin_factory=factory)

def rotateMax():
	print("Go to max")
	servo.max()
	sleep(1)
	
	
def rotateMin():
	print("Go to min")
	servo.min()
	sleep(1)
