from bs4 import BeautifulSoup


class getCleanData:

    def clear_text(self, text):
        # original_text = text

        # Remove "" at the beggining and at the end
        if text[0] == '"':
            text = text[1:]

        if text[-1] == '"':
            text = text[:-1]

        # Create a BeautifulSoup object from the HTML
        soup = BeautifulSoup(text, 'html.parser', )

        # Extract only the text values with spaces between list items (line 90)
        text = soup.get_text(separator=" ")

        # Delete words longer than 50 characters as it is most likely some code giving only small or none value to the context (line 69)
        words = text.split()
        filtered_words = [word for word in words if len(word) <= 50]
        text = ' '.join(filtered_words)

        # IF there is no text, it is perhaps better to take at least html tags, more info about the task needed (line 87)
        # if text == "":
        #     text = original_text

        return text

    def get_all_texts(self, filename):
        # Get all texts, clean them and prepare for embedding

        all_texts = {} # Resulting documents
        unique_check_text = set() # Check for duplicities

        # Open the file for reading
        with open(filename, 'r') as file:

            # Iterate over each line in the file
            for line_num, line in enumerate(file, 1):

                # Skip first line
                if line_num == 1:
                    continue

                # Split the line into number and text
                try:
                    doc_id, text = line.strip().split('\t', 1)
                except:
                    print("Line %d with id %s cannot be split: " % (line_num, line))
                    continue

                # Convert the number to an integer
                doc_id = int(doc_id)

                # Clearing texts from html tags, etc
                text = self.clear_text(text)

                # Remove duplicates based on text
                if text in unique_check_text or text == "":
                    # print("Repeating:", line_num, text)
                    continue

                if doc_id in all_texts:
                    # Append duplicated IDs texts
                    all_texts[doc_id] += " " + text
                else:
                    all_texts[doc_id] = text


                unique_check_text.add(text)
                #print(line_num, ' -- ',text)

        # Convert dict to list
        all_texts = [[k, v] for k, v in all_texts.items()]

        return all_texts
