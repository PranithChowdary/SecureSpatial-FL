import hashlib
import time

class BlockchainAuth:
    """
    Simulated Blockchain-Based Remote Mutual Authentication (B-RMA) Protocol.
    """
    def __init__(self):
        self.ledger = {}  # Node_ID: Node_Metadata
        self.authorized_nodes = set()
        
    def register_node(self, node_id, public_key):
        """
        Register a node in the blockchain ledger.
        """
        timestamp = time.time()
        registration_token = self._generate_token(node_id, public_key, timestamp)
        self.ledger[node_id] = {
            'public_key': public_key,
            'token': registration_token,
            'status': 'Registered'
        }
        print(f"Node {node_id} registered with token: {registration_token[:10]}...")
        return registration_token

    def authenticate_node(self, node_id, token):
        """
        Authenticate a node before participating in FL.
        """
        if node_id in self.ledger and self.ledger[node_id]['token'] == token:
            self.authorized_nodes.add(node_id)
            print(f"Node {node_id} authenticated successfully.")
            return True
        else:
            print(f"Authentication failed for Node {node_id}.")
            return False

    def is_authorized(self, node_id):
        return node_id in self.authorized_nodes

    def _generate_token(self, node_id, public_key, timestamp):
        data = f"{node_id}{public_key}{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()

if __name__ == "__main__":
    b_auth = BlockchainAuth()
    node_id = "node_01"
    pub_key = "PUB_KEY_01"
    
    token = b_auth.register_node(node_id, pub_key)
    if b_auth.authenticate_node(node_id, token):
        print("Success: Node is authorized to participate.")
    else:
        print("Error: Node is not authorized.")
