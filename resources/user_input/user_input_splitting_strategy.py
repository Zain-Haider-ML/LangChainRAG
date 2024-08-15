def display_menu():
    print("Please select one of the following options:")
    print("1. SentenceTransformersTokenTextSplitter")
    print("2. TokenTextSplitter")
    print("3. RecursiveCharacterTextSplitter")
    print("4. Custom")
    print("5. CharacterTextSplitter")

def get_user_choice():
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if choice in range(1, 5):
                return choice
            else:
                print("Invalid choice. Please select a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main_menu():
    display_menu()
    user_choice = get_user_choice()

    if user_choice == 1:
        ss = 'SentenceTransformersTokenTextSplitter'
    elif user_choice == 2:
        ss = 'TokenTextSplitter'
    elif user_choice == 3:
        ss = 'RecursiveCharacterTextSplitter'
    elif user_choice == 4:
        ss = 'CharacterTextSplitter'
    
    print(f"You selected Option {user_choice}")
    return ss