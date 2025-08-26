from cryptography.fernet import Fernet
import os

class SecurityManager:
    def __init__(self):
        self.cipher_key = os.getenv('CIPHER_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.cipher_key)

    def encrypt_sensitive_data(self, data: str) -> bytes:
        """Encrypt sensitive data before storage"""
        return self.cipher.encrypt(data.encode())

    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data after retrieval"""
        return self.cipher.decrypt(encrypted_data).decode()