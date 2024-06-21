int countChar(const char* str, char c);

int main(){
    char* str = "Terezka je velmi sikovna";

    printf("%d", countChar(str, 'i'));
}

//Spočíta počet výskytov znaku v str
int countChar(const char* str, char c){
    int count = 0;
    for(int i = 0; str[i] != '\0'; i++){
        if(str[i] == c){
            count++;
        }
    }
    return count;
}