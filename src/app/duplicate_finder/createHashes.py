import binascii
import pickle
import random
import time
from zipfile import ZipFile

"""
A MinHash-based near duplicate detection for the WLO dataset.

(based on a tutorial on MinHash https://github.com/chrisjmccormick/MinHash)
"""


def create_hashes(data_folder: str = "data"):
    file = f"{data_folder}/wirlernenonline3-dedup.txt.zip"

    # open the zip file in read mode
    print(file)
    with ZipFile(file, "r") as zip_file:
        zip_file.extractall(data_folder)

    numHashes = 100
    dataFile = f"{data_folder}/wirlernenonline3-dedup.txt"

    def shingleWords(words):
        # 'shinglesInDoc' will hold all of the unique shingle IDs present in the
        # current document. If a shingle ID occurs multiple times in the document,
        # it will only appear once in the set (this is a property of Python sets).
        words = [x.replace("\n", "") for x in words if x]

        shinglesInDoc = set()
        # For each word in the document...
        for index in range(0, len(words) - 2):
            # Construct the shingle text by combining three words together.
            shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]

            # Hash the shingle to a 32-bit integer.
            crc = binascii.crc32(shingle.encode()) & 0xFFFFFFFF

            # Add the hash value to the list of shingles for the current document.
            # Note that set objects will only add the value to the set if the set
            # doesn't already contain it.
            shinglesInDoc.add(crc)
        if len(words) <= 2:
            shingle = " ".join(words)
            crc = binascii.crc32(shingle.encode()) & 0xFFFFFFFF
            shinglesInDoc.add(crc)
        return shinglesInDoc

    # =============================================================================
    #               Convert Documents To Sets of Shingles
    # =============================================================================

    print("Shingling articles...", flush=True)

    # The current shingle ID value to assign to the next new shingle we
    # encounter. When a shingle gets added to the dictionary, we'll increment this
    # value.
    # curShingleID = 0

    # Create a dictionary of the articles, mapping the article identifier (e.g.,
    # "t8470") to the list of shingle IDs that appear in the document.
    docsAsShingleSets = {}

    docs = {}
    docNames = []
    docUrls = {}

    t0 = time.time()

    totalShingles = 0

    cnt = 0
    with open(dataFile) as inf:
        fit = iter(inf)
        for line in fit:

            # Read all of the words (they are all on one line) and split them by white
            # space.
            # line = f.readline()
            words = line.strip().split(" ")

            # Retrieve the article ID, which is the first word on the line.
            docID = words[0]

            # Maintain a list of all document IDs.
            docNames.append(docID)

            docUrl = words[1]
            if docUrl not in docUrls.keys():
                docUrls[docUrl] = []
            if docUrl != "_":
                docUrls[docUrl].append(docID)

            del words[0]
            del words[0]

            # 'shinglesInDoc' will hold all of the unique shingle IDs present in the
            # current document. If a shingle ID occurs multiple times in the document,
            # it will only appear once in the set (this is a property of Python sets).
            shinglesInDoc = shingleWords(words)

            # Store the completed list of shingles for this document in the dictionary.
            docsAsShingleSets[docID] = shinglesInDoc
            docs[docID] = words

            # Count the number of shingles across all documents.
            totalShingles = totalShingles + (len(words) - 2)

            cnt += 1

    numDocs = cnt

    # Report how long shingling took.
    print("\nShingling " + str(numDocs) + " docs took %.2f sec." % (time.time() - t0))

    print("\nAverage shingles per doc: %.2f" % (totalShingles / numDocs))

    # =============================================================================
    #                 Generate MinHash Signatures
    # =============================================================================

    # Time this step.
    t0 = time.time()

    print("\nGenerating random hash functions...")

    # Record the maximum shingle ID that we assigned.
    maxShingleID = 2**32 - 1

    # We need the next largest prime number above 'maxShingleID'.
    # I looked this value up here:
    # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    nextPrime = 4294967311

    # Our random hash function will take the form of:
    #   h(x) = (a*x + b) % c
    # Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
    # a prime number just greater than maxShingleID.

    # Generate a list of 'k' random coefficients for the random hash functions,
    # while ensuring that the same value does not appear multiple times in the
    # list.
    def pickRandomCoeffs(k):
        # Create a list of 'k' random values.
        randList = []

        while k > 0:
            # Get a random shingle ID.
            randIndex = random.randint(0, maxShingleID)

            # Ensure that each random number is unique.
            while randIndex in randList:
                randIndex = random.randint(0, maxShingleID)

                # Add the random number to the list.
            randList.append(randIndex)
            k = k - 1

        return randList

    # For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
    coeffA = pickRandomCoeffs(numHashes)
    coeffB = pickRandomCoeffs(numHashes)

    def getSignature(shingleIDSet):
        signature = []

        # For each of the random hash functions...
        for i in range(0, numHashes):

            # For each of the shingles actually in the document, calculate its hash code
            # using hash function 'i'.

            # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
            # the maximum possible value output by the hash.
            minHashCode = nextPrime + 1

            # For each shingle in the document...
            for shingleID in shingleIDSet:
                # Evaluate the hash function.
                hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

                # Track the lowest hash code seen.
                if hashCode < minHashCode:
                    minHashCode = hashCode

            # Add the smallest hash code value as component number 'i' of the signature.
            signature.append(minHashCode)
        return signature

    print("\nGenerating MinHash signatures for all documents...")

    # List of documents represented as signature vectors
    signatures = []

    # Rather than generating a random permutation of all possible shingles,
    # we'll just hash the IDs of the shingles that are *actually in the document*,
    # then take the lowest resulting hash code value. This corresponds to the index
    # of the first shingle that you would have encountered in the random order.
    cnt = 0
    # For each document...
    for docID in docNames:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt, flush=True)
        # Get the shingle set for this document.
        shingleIDSet = docsAsShingleSets[docID]

        # The resulting minhash signature for this document.
        signature = getSignature(shingleIDSet)

        # Store the MinHash signature for this document.
        signatures.append(signature)

    # Calculate the elapsed time (in seconds)
    elapsed = time.time() - t0

    print("\nGenerating MinHash signatures took %.2fsec" % elapsed)

    print("\nStoring data ...")

    with open(f"{data_folder}/hashes.p", "wb") as f:
        pickle.dump(signatures, f)

    with open(f"{data_folder}/docnames.p", "wb") as f:
        pickle.dump(docNames, f)

    with open(f"{data_folder}/coeffa.p", "wb") as f:
        pickle.dump(coeffA, f)

    with open(f"{data_folder}/coeffb.p", "wb") as f:
        pickle.dump(coeffB, f)

    with open(f"{data_folder}/docs.p", "wb") as f:
        pickle.dump(docs, f)

    with open(f"{data_folder}/docUrls.p", "wb") as f:
        pickle.dump(docUrls, f)

    with open(f"{data_folder}/shingles.p", "wb") as f:
        pickle.dump(docsAsShingleSets, f)

    print("\n... done!")


if __name__ == "__main__":
    data = "../../data"
    create_hashes(data)
