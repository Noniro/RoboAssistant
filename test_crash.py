import time
from main_ui import RobotSupervisorApp

print("Starting RobotSupervisorApp test...")
try:
    app = RobotSupervisorApp()
    print("App initialized. Simulating 'Initialize System' click in 1 second...")
    app.after(1000, lambda: app.toggle_system())
    app.after(3000, lambda: app.destroy())
    app.mainloop()
    print("Test finished successfully without crashing.")
except Exception as e:
    import traceback
    print("CRASH DETECTED!")
    traceback.print_exc()
