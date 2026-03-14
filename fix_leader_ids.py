import scservo_sdk as scs
import time

PORT = 'COM5'
BAUDRATE = 1000000

portHandler = scs.PortHandler(PORT)
packetHandler = scs.PacketHandler(0)

def change_id(old_id, new_id):
    print(f"Changing motor ID {old_id} to {new_id}...")
    
    # Write to Address 48 (Lock) with value 0 to unlock EEPROM
    packetHandler.write1ByteTxRx(portHandler, old_id, 48, 0)
    time.sleep(0.2)
    
    # Write to Address 5 (ID) with the new ID
    packetHandler.write1ByteTxRx(portHandler, old_id, 5, new_id)
    time.sleep(0.2)
    
    # Write to Address 48 (Lock) with value 1 to lock EEPROM 
    # (We use the new_id now, because the motor responds to its new ID!)
    packetHandler.write1ByteTxRx(portHandler, new_id, 48, 1)
    time.sleep(0.2)
    
    print(f"Verified modification to {new_id}.")

if __name__ == "__main__":
    if portHandler.openPort() and portHandler.setBaudRate(BAUDRATE):
        print(f"Successfully opened {PORT} at {BAUDRATE} baud.\n")
        
        print("Starting the ID Swap between Motor 4 and Motor 6 (Wrist Flex and Gripper)...")
        print("Note: We must use a temporary ID (20) so they don't overlap into the same ID.\n")
        
        try:
            # 1. Change Motor 4 to 20
            change_id(4, 20)
            
            # 2. Change Motor 6 to 4
            change_id(6, 4)
            
            # 3. Change Motor 20 to 6
            change_id(20, 6)
            
            print("\nSWAP COMPLETE! Motor 4 and 6 have been swapped.")
            print("Please run the lerobot-calibrate script again.")
        except Exception as e:
            print(f"An error occurred during remapping: {e}")
            
        portHandler.closePort()
    else:
        print(f"Failed to open {PORT}. Is another application using it?")
