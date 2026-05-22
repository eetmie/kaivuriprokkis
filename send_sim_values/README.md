# UDP random value demo

Run the server in one terminal:

```bash
python -m send_sim_values.server
```

Run the client in another terminal:

```bash
python -m send_sim_values.client
```

The server listens on `127.0.0.1:5005` and sends three random float values once per second.
The client receives those values and prints the newest packet.
