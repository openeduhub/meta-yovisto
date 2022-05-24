import binascii
import pickle
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""
A MinHash-based near duplicate detection for the WLO dataset.

(based on a tutorial on MinHash https://github.com/chrisjmccormick/MinHash)
"""


class DuplicateFinder:
    docNames, docs, signatures, arr, coeffA, coeffB = None, None, None, None, None, None
    docsAsShingleSets, docUrls = None, None

    numHashes = 100

    def __init__(self, file_path: str = "data"):
        with open(f"{file_path}/hashes.p", "rb") as f:
            self.signatures = pickle.load(f)

        with open(f"{file_path}/docnames.p", "rb") as f:
            self.docNames = pickle.load(f)

        with open(f"{file_path}/docs.p", "rb") as f:
            self.docs = pickle.load(f)

        with open(f"{file_path}/docUrls.p", "rb") as f:
            self.docUrls = pickle.load(f)

        with open(f"{file_path}/coeffa.p", "rb") as f:
            self.coeffA = pickle.load(f)

        with open(f"{file_path}/coeffb.p", "rb") as f:
            self.coeffB = pickle.load(f)

        # 		with open('./data/shingles.p', 'rb') as f:
        # 			self.docsAsShingleSets=pickle.load(f)

        print(self.docs[self.docNames[5]])

        self.arr = np.array(self.signatures)

    def shingleWords(self, words):  # has to be the same method as for hash creation
        shinglesInDoc = set()
        words = [x.replace("\n", "") for x in words if x]
        for index in range(0, len(words) - 2):
            shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]
            crc = binascii.crc32(shingle.encode()) & 0xFFFFFFFF
            shinglesInDoc.add(crc)
        return shinglesInDoc

    def getSignature(
        self, shingleIDSet
    ):  # has to be the same method as for hash creation
        signature = []
        nextPrime = 4294967311

        for i in range(0, self.numHashes):
            minHashCode = nextPrime + 1

            for shingleID in shingleIDSet:
                hashCode = (self.coeffA[i] * shingleID + self.coeffB[i]) % nextPrime

                if hashCode < minHashCode:
                    minHashCode = hashCode

            signature.append(minHashCode)
        return signature

    def runById(self, d):
        if d in self.docNames:
            return [[d, "1.0", " ".join(self.docs[d])]]
        return []

    def runByUrl(self, url):
        result = []
        if url in self.docUrls.keys():
            for d in self.docUrls[url]:
                result.append([d, "1.0", " ".join(self.docs[d])])
        return result

    def runByText(self, text, threshold):
        shingles = self.shingleWords(text.split())
        sig = self.getSignature(shingles)

        dists = cosine_similarity(self.arr, [sig])
        sorted_arg = np.argsort(dists.ravel())

        closest = reversed(sorted_arg[-10:])
        # print(sorted_arg[-10:])
        result = []

        for d in closest:
            # print (d, dists[d])
            # print (self.docNames[d])
            # print (" ".join(self.docs[self.docNames[d]]))
            # print ("----")
            if dists[d][0] > threshold:
                result.append(
                    [
                        self.docNames[d],
                        dists[d][0],
                        " ".join(self.docs[self.docNames[d]]),
                    ]
                )

        return result


if __name__ == "__main__":

    text = sys.argv[1]

    print("Searching near duplicates for: '" + text + "'")

    p = DuplicateFinder()
    print("TEXT")
    for r in p.runByText(text):
        print(r)

    print("ID")
    for r in p.runById(text):
        print(r)

    print("URL")
    for r in p.runByUrl(text):
        print(r)
