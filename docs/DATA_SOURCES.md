# Bilgisayar Bilimleri Veri Kaynakları (Data Sources)

Mergen-SNN'in Aşama 3 ve devamında eğitileceği Bilgisayar Bilimleri (Computer Science) korpusu için belirlenen açık erişimli ve lisanslı kaynaklar aşağıda listelenmiştir.

---

## Veri Kaynakları Tablosu

| Kaynak Adı | Konu / Alan | URL | Lisans / Durum | Yerel Hedef Klasör |
| :--- | :--- | :--- | :--- | :--- |
| **Python Official Documentation** | Dil & Standart Kütüphane | `https://docs.python.org/3/` | PSF License | `data/raw/cs_corpus/python_docs/` |
| **GNU C Reference Manual & C Docs** | C Dili & Sistem Programlama | `https://www.gnu.org/software/gnu-c-manual/` | GNU FDL | `data/raw/cs_corpus/c_docs/` |
| **cppreference (C++ Standard)** | C++ Bellek Modeli & STL | `https://en.cppreference.com/` | CC-BY-SA 3.0 | `data/raw/cs_corpus/cpp_docs/` |
| **Cython Official Documentation** | C-Extensions & Performans | `https://cython.readthedocs.io/en/latest/` | Apache 2.0 | `data/raw/cs_corpus/cython_docs/` |
| **Operating Systems: Three Easy Pieces (OSTEP)** | İşletim Sistemleri (CPU, Bellek, Kalıcılık) | `https://pages.cs.wisc.edu/~remzi/OSTEP/` | Free Online (Remzi & Andrea Arpaci-Dusseau) | `data/raw/cs_corpus/ostep/` |
| **Computer Organization and Design (Patterson & Hennessy)** | Bilgisayar Mimarisi & RISC-V/MIPS | Referans Notları / Kamuya Açık Özetler | Eğitim Amaçlı Referans | `data/raw/cs_corpus/architecture/` |
| **Computer Science from the Bottom Up** | İşletim Sistemi & Düşük Seviye Mimari | `https://www.bottomupcs.com/` | GNU FDL (Ian Wienand) | `data/raw/cs_corpus/systems/` |
| **Dive into Systems** | Sistem Mimarisi, Derleyiciler, C | `https://diveintosystems.org/` | CC-BY-NC-ND 4.0 | `data/raw/cs_corpus/systems/` |
| **Garbage Collection Klasik Makaleleri** | Bellek Yönetimi & GC Algoritmaları | `https://github.com/papers-we-love/papers-we-love/tree/master/garbage-collection` | Açık Akademik / PWL | `data/raw/cs_corpus/gc_papers/` |
| **The UNIX Time-Sharing System (Ritchie & Thompson)** | Unix Tasarım Felsefesi & Çekirdek | `https://www.bell-labs.com/usr/dmr/www/` | Tarihi Belge / Bell Labs | `data/raw/cs_corpus/unix_os/` |

---

## Veri İşleme Prensipleri

1. **Ham Metin Çıkarımı:** HTML/PDF kaynaklarından sadece düz metin (plain text) ve kod blokları ayıklanır; biçimlendirme etiketleri temizlenir.
2. **Karakter ve Alt-Kelime Tokenizasyonu:** İlgili metinler `data/processed/` dizini altında tokenize edilmiş biçimde saklanır.
3. **Gizlilik ve Boyut:** Ham korpus dosyaları `.gitignore` kapsamında yer alır ve Git deposuna eklenmez.
