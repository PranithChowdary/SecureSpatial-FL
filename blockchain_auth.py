import hashlib
import time
import secrets

class BlockchainAuth:
    """
    Enhanced B-RMA Protocol with Challenge-Response and BAN Logic hooks.
    Ensures that only authorized IoT nodes participate in FedAvg.
    """
    def __init__(self):
        self.ledger = {}  # Node_ID: {public_key, last_nonce, status}
        self.authorized_sessions = {} # Node_ID: session_expiry
        
    def register_node(self, node_id, public_key):
        """
        Phase 1: Registration. 
        In BAN Logic: 'Trust Gate' establishes that the Server believes in Node's PK[cite: 84].
        """
        self.ledger[node_id] = {
            'public_key': public_key,
            'status': 'Registered',
            'last_nonce': None
        }
        print(f"Blockchain Ledger Updated: Node {node_id} is now a trusted participant.")

    def initiate_challenge(self, node_id):
        """
        Phase 2: Mutual Authentication (Challenge).
        Prevents Replay Attacks by generating a unique cryptographic Nonce.
        """
        if node_id not in self.ledger:
            return None
        
        # Generate a random 32-byte nonce
        nonce = secrets.token_hex(32)
        self.ledger[node_id]['last_nonce'] = nonce
        return nonce

    def verify_response(self, node_id, signed_nonce):
        """
        Phase 3: Verification.
        BAN Logic Proof: If Server sees {Nonce} signed by PK, Server believes Node is 'Fresh'.
        """
        if node_id not in self.ledger:
            return False
            
        expected_nonce = self.ledger[node_id]['last_nonce']
        
        # In a real system, you'd verify the digital signature here.
        # Simulation: We check if the response contains the hash of the nonce + PK.
        valid_signature = self._simulate_sig_check(node_id, expected_nonce, signed_nonce)
        
        if valid_signature:
            # Session valid for 1 hour (3600 seconds)
            self.authorized_sessions[node_id] = time.time() + 3600
            print(f"B-RMA Success: Node {node_id} authorized for current FL round.")
            return True
        
        print(f"Security Alert: Authentication failed for {node_id}!")
        return False

    def is_authorized(self, node_id):
        """Checks if the session is still fresh/valid."""
        if node_id not in self.authorized_sessions:
            return False
        return time.time() < self.authorized_sessions[node_id]

    def _simulate_sig_check(self, node_id, nonce, signature):
        """Simulates RSA/ECC signature verification."""
        pk = self.ledger[node_id]['public_key']
        expected_sig = hashlib.sha256(f"{nonce}{pk}".encode()).hexdigest()
        return signature == expected_sig

if __name__ == "__main__":
    # Example Workflow for the Research Paper
    b_auth = BlockchainAuth()
    client_id = "ESP32_Node_01"
    
    # 1. Registration
    b_auth.register_node(client_id, "ECC_PUBLIC_KEY_AUM_S00462031")
    
    # 2. Challenge-Response (The B-RMA "Trust Gate" [cite: 84])
    challenge = b_auth.initiate_challenge(client_id)
    
    # Client 'signs' the challenge
    mock_sig = hashlib.sha256(f"{challenge}ECC_PUBLIC_KEY_AUM_S00462031".encode()).hexdigest()
    
    # 3. Final Authorization
    if b_auth.verify_response(client_id, mock_sig):
        print("Protocol Verified: Node is authorized to upload weights.")