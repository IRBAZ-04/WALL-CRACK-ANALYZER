"""
launcher.py
------------
Single unified interactive menu launcher for the entire Wall Crack Analyzer project.
"""
import os
import sys
import subprocess

PYTHON = sys.executable

def main():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 65)
        print("    AUTOMATIC WALL CRACK ANALYZER (UCSC313L PROJECT)")
        print("       100% From-Scratch Algorithms (Zero OpenCV)")
        print("=" * 65)
        print("  [1] Launch Modern Desktop GUI (PySide6 - Recommended)")
        print("  [2] Launch Classic GUI (Tkinter)")
        print("  [3] Analyze Cracked Wall Photo (Campus Academic Block)")
        print("  [4] Screen Clean Sound Wall (Library Reading Room)")
        print("  [5] Run Fast Classical Pipeline Only (--no-ai)")
        print("  [6] Run Automated 6-Point Verification Test Suite")
        print("  [7] Run Full Algorithm Benchmark Matrix & Charts")
        print("  [0] Exit")
        print("=" * 65)
        choice = input("Select an option (0-7): ").strip()

        if choice == "1":
            subprocess.call([PYTHON, "gui/app_pyside.py"])
        elif choice == "2":
            subprocess.call([PYTHON, "gui/app.py"])
        elif choice == "3":
            subprocess.call([PYTHON, "main.py", "dataset/raw/CAMPUS_ACADEMIC_001.jpg", "canny"])
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            subprocess.call([PYTHON, "main.py", "dataset/raw/CAMPUS_CLEAN_007.jpg", "canny"])
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            subprocess.call([PYTHON, "main.py", "dataset/raw/CAMPUS_ACADEMIC_001.jpg", "canny", "--no-ai"])
            input("\nPress Enter to return to menu...")
        elif choice == "6":
            subprocess.call([PYTHON, "evaluation/test_ai_integration.py"])
            input("\nPress Enter to return to menu...")
        elif choice == "7":
            subprocess.call([PYTHON, "evaluation/comparison.py"])
            input("\nPress Enter to return to menu...")
        elif choice == "0":
            print("\nExiting. Good luck with your project presentation!")
            break
        else:
            print("\nInvalid choice. Please select 0 to 7.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
