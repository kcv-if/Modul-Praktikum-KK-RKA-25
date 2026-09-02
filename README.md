# Modul Praktikum Logical Agents

Modul ini membahas agent yang mengambil keputusan dengan **menalar**, bukan
dengan mencocokkan situasi ke tabel aksi. Agent menyimpan apa yang diketahuinya
dalam sebuah knowledge base, lalu menurunkan kesimpulan baru dari situ. Contoh
yang dipakai sepanjang modul adalah Wumpus World: sebuah gua yang isinya tidak
terlihat, sehingga agent harus menyimpulkan kotak mana yang aman hanya dari bau
dan hembusan angin yang dirasakannya.

## Alur materi

Empat notebook, dikerjakan berurutan. Notebook berikutnya memakai konsep dan
fungsi yang sudah diperkenalkan sebelumnya, jadi jangan dilompati.

**`01 - KB & Wumpus World.ipynb`**
Apa itu knowledge-based agent dan kenapa memisahkan pengetahuan dari program itu
menguntungkan. Operasi TELL dan ASK. Aturan main Wumpus World: apa yang membunuh
agent, apa yang bisa dirasakannya, dan bagaimana skornya dihitung. Diakhiri
dengan menelusuri langkah demi langkah bagaimana agent menyimpulkan letak wumpus
dari kotak yang belum pernah dikunjunginya.

**`02 - Entailment & Inference.ipynb`**
Bedanya syntax, semantics, dan model. Lalu inti dari seluruh modul ini:
**entailment**, yaitu kapan sebuah kalimat pasti mengikuti dari kalimat lain.
Idenya diperagakan dengan mengenumerasi delapan kemungkinan dunia dan melihat di
dunia mana saja pengetahuan agent bisa benar. Di sini juga dibahas dua sifat
algoritma inferensi, sound dan complete.

**`03 - Propositional Logic.ipynb`**
Bahasa formal untuk menuliskan pengetahuan tadi. Simbol proposisi, lima konektif
logika, dan truth table. Bagian yang paling sering bikin bingung ada di sini:
kenapa "5 is even implies Sam is smart" bernilai benar padahal Sam belum tentu
pintar.

**`04 - Model Checking.ipynb`**
Wumpus World ditulis ulang sebagai lima aturan logika, lalu pertanyaan "apakah
ada lubang di [1,2]" dijawab otomatis dengan memeriksa 128 kemungkinan dunia.
Ditutup dengan alasan kenapa cara ini tidak bisa dipakai untuk dunia yang lebih
besar, yang jadi jembatan ke materi berikutnya.

## Menjalankan notebook

**Google Colab.** Buka notebooknya, lalu jalankan sel Setup paling atas. Tidak
perlu install apa pun.

**Lokal.**

```bash
git clone https://github.com/kcv-if/Modul-Praktikum-KK-RKA-25.git
cd Modul-Praktikum-KK-RKA-25
pip install -r requirements.txt
jupyter lab
```

Buka notebook dari dalam folder repo, lalu jalankan sel Setup paling atas.

Environment sudah siap kalau output terakhir sel Setup mencetak:

```
Check       : tt_entails(P & Q, Q) = True
```

## Cara mengerjakan

Tiap sub-topik punya dua bagian dengan peran berbeda.

**Penjelasan** berisi konsepnya. Tidak ada kode di sini. Baca sampai paham dulu,
jangan langsung lompat ke cell kode di bawahnya.

**Contoh penerapan** berisi kode yang benar-benar bisa dijalankan. Kodenya bukan
tempelan: hampir semuanya membangkitkan ulang gambar yang ada di slide dan buku.
Jalankan, lalu cocokkan angkanya dengan gambar tersebut. Kalau jumlah barisnya
cocok, berarti kamu membaca gambarnya dengan benar.

**Latihan Soal** ada di bagian paling bawah tiap notebook. Semua soal
dikumpulkan di sana, tidak ada yang diselipkan di tengah materi. Tiap soal
disediakan cell kosong untuk jawabanmu, dan pembahasannya ada di balik blok yang
bisa dibuka-tutup.

Sebelum mengumpulkan, jalankan **Restart and Run All** dan pastikan tidak ada
error dari atas sampai bawah.
