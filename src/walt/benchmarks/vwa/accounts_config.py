# Multi-account configuration for shopping site authentication
# This eliminates session conflicts by using unique accounts for each auth instance

CUSTOMER_ACCOUNTS = [
    {"email": "convexegg@gmail.com", "password": "Password.123"},
    {"email": "john.smith.xyz@gmail.com", "password": "Password.123"},
    {"email": "jane.doe@hotmail.com", "password": "Password.123"},
    {"email": "bbjones@gmail.com", "password": "Password.123"},
    {"email": "helloworld@yahoo.com", "password": "Password.123"},
    {"email": "jla_7781@gmail.com", "password": "Password.123"},
    {"email": "bob123@hotmail.com", "password": "Password.123"},
    {"email": "marym@gmail.com", "password": "Password.123"},
    {"email": "john.lee@yahoo.com", "password": "Password.123"},
    {"email": "janesmith@gmail.com", "password": "Password.123"},
    {"email": "daniel.jackson@hotmail.com", "password": "Password.123"},
    {"email": "lisa.kim@gmail.com", "password": "Password.123"},
    {"email": "matt.baker@yahoo.com", "password": "Password.123"},
    {"email": "johndoe123@gmail.com", "password": "Password.123"},
    {"email": "janesmith456@yahoo.com", "password": "Password.123"},
    {"email": "coolcat321@hotmail.com", "password": "Password.123"},
    {"email": "harrypotterfan1@gmail.com", "password": "Password.123"},
    {"email": "avidreader99@yahoo.com", "password": "Password.123"},
    {"email": "artsygal123@hotmail.com", "password": "Password.123"},
    {"email": "soccerfanatic22@gmail.com", "password": "Password.123"},
    {"email": "beachlover99@yahoo.com", "password": "Password.123"},
    {"email": "fashionista88@gmail.com", "password": "Password.123"},
    {"email": "fitnessjunkie22@yahoo.com", "password": "Password.123"},
    {"email": "musiclover99@hotmail.com", "password": "Password.123"},
    {"email": "gamingpro456@gmail.com", "password": "Password.123"},
    {"email": "xyz@gmail.com", "password": "Password.123"},
    {"email": "emma.lopez@gmail.com", "password": "Password.123"},
]

REDDIT_ACCOUNTS = [
    {"email": "EpicThunderCat", "password": "test1234"},
    {"email": "VeryVeryVeni", "password": "test1234"},
    {"email": "BallPointedPeen", "password": "test1234"},
    {"email": "houselegs", "password": "test1234"},
    {"email": "AnnaDawsonArt", "password": "test1234"},
    {"email": "anasse_", "password": "test1234"},
    {"email": "alternateartreality", "password": "test1234"},
    {"email": "suffocatmeow", "password": "test1234"},
    {"email": "csvwart", "password": "test1234"},
    {"email": "babbittybabbitt", "password": "test1234"},
    {"email": "EllieMacBeth", "password": "test1234"},
    {"email": "BensDrawings", "password": "test1234"},
    {"email": "jackhendsbee", "password": "test1234"},
    {"email": "AlexSlapnuts", "password": "test1234"},
    {"email": "artofmeh", "password": "test1234"},
    {"email": "Nono090304", "password": "test1234"},
    {"email": "ninadrawsalot", "password": "test1234"},
    {"email": "Rehd96", "password": "test1234"},
    {"email": "sasquatchinheat", "password": "test1234"},
    {"email": "dropsandbits", "password": "test1234"},
    {"email": "Everiet", "password": "test1234"},
    {"email": "Top_Entertainer_760", "password": "test1234"},
    {"email": "kUkara4", "password": "test1234"},
    {"email": "ssteiny", "password": "test1234"},
    {"email": "purr_in_ink", "password": "test1234"},
    {"email": "meprobamatedowned", "password": "test1234"},
    {"email": "franciscoaldeao", "password": "test1234"},
    {"email": "irsin", "password": "test1234"},
    {"email": "C0ncL", "password": "test1234"},
    {"email": "sureMOEDesign", "password": "test1234"},
    {"email": "Dylan_Kowalski_3D", "password": "test1234"},
    {"email": "marvelsgrantman136", "password": "test1234"},
]


def get_account_by_index(index: int) -> dict:
    """Get account by index with round-robin cycling."""
    return CUSTOMER_ACCOUNTS[index % len(CUSTOMER_ACCOUNTS)]


def get_total_accounts() -> int:
    """Get total number of available accounts."""
    return len(CUSTOMER_ACCOUNTS)
