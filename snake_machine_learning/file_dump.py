import pickle

class FileDump:
    def save_network(self, network):
        with open('network.bin', 'wb') as outp:
            pickle.dump(network, outp)

    def load_network(self):
        try:
            with open('network.bin', 'rb') as inp:
                network = pickle.load(inp)
                return network
        except:
            return None
