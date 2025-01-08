from typing import *
import os
from gec_dataset import IGECDataset, Entry
import xml.etree.ElementTree as ET

class FceDataset(IGECDataset):

    def __init__(self, fce_dir: str) -> None:

        self._entries: List[Entry] = []

       
        for dir in os.listdir("datasets/fce"):
        # print("DIR IS ---------------------",dir)
            if dir == ".DS_Store":
                continue
            for file in os.listdir(os.path.join("datasets/fce", dir)):
                if file == ".DS_Store":
                    continue
                # print("FILE IS ---------------------",file)         
                with open(os.path.join("datasets/fce",dir ,file)) as f:
                    
                    tree = ET.parse(f)
                    root = tree.getroot()
                                  
                    for paragraph in root.iter('p'):
                        ful_t = ET.tostring(paragraph , encoding='unicode')
                        t = ful_t.replace("<p>",'').replace('</p>','')
                        # print(ful_t)
                        NS_text={}
                        for i in range(len(paragraph)):
                            NS_text[i]= {}
                            for ii in range(len(paragraph[i])):
                                NS_text[i][paragraph[i][ii].tag] = paragraph[i][ii].text
                                
                        # print(NS_text)
                        
                        cor_sen=""
                        ori_sen=""
                        NS_count=0
                        flag=0
                        while( len(t)>0):
                            
                            if( t.find("<NS") == -1) :
                                cor_sen += t
                                ori_sen += t
                                t=[]
                                break
                            cor_sen += t[:t.find("<NS")]
                            ori_sen += t[:t.find("<NS")]
                            t= t[t.find("<NS")+1:]  

                        

                            if ('c' in NS_text[NS_count]):
                                if  NS_text[NS_count]['c']!= None:
                                    if (cor_sen != ""):
                                        cor_sen +=' '
                                    cor_sen+= NS_text[NS_count]['c']

                            if ('i' in NS_text[NS_count]):
                                if  NS_text[NS_count]['i']!= None:
                                    if (ori_sen != ""):
                                        ori_sen +=' '
                                    ori_sen+= NS_text[NS_count]['i']
                            
                            NS_count+=1

                            if (t.find("<NS") < t.find("</NS>") and t.find("<NS")!= -1 and t.find("</NS>") != -1  ):
                                flag =1
                                t= t[t.find("<NS")+1:] 
                                t=t[t.find("</NS>")+5:]
                                break

                            t=t[t.find("</NS>")+5:]

                        if(flag == 0):
                            self._entries.append(
                                Entry(
                                    original_sentence=ori_sen,
                                    corrected_sentence=cor_sen
                                ))
                    
                        

    def __len__(self):
        """
        Return total number of entries of the dataset
        """
        return len(self._entries)

    def __getitem__(self, i: int) -> Entry:
        """
        Return i-th entry of the dataset
        """
        return self._entries[i]


if __name__ == "__main__":

    dataset = FceDataset("datasets/fce")

    print(f"Number of entries: {len(dataset)}")
    for i in range(3):
        print(f"Entry[{i}]: {dataset[i]}")
