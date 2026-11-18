"""Self-signed certificate so a phone browser may open the camera (browsers require HTTPS for camera access off localhost)."""
import os, datetime, ipaddress

def ensure_cert(directory, ip):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    os.makedirs(directory, exist_ok=True)
    crt, key, tag = (os.path.join(directory, n) for n in ("phone.crt", "phone.key", "phone.ip"))
    if os.path.exists(crt) and os.path.exists(key) and os.path.exists(tag) and open(tag).read() == ip:
        return crt, key
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vingage-local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try: san.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except ValueError: pass
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=365))
         .add_extension(x509.SubjectAlternativeName(san), critical=False).sign(k, hashes.SHA256()))
    open(crt, "wb").write(c.public_bytes(serialization.Encoding.PEM))
    open(key, "wb").write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    open(tag, "w").write(ip)
    return crt, key
