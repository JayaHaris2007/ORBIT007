import { useState } from 'react';

const quotes = [
    { text: "Wake up to reality! Nothing ever goes as planned in this accursed world.", char: "Madara Uchiha", anime: "Naruto" },
    { text: "If you don't take risks, you can't create a future.", char: "Monkey D. Luffy", anime: "One Piece" },
    { text: "Push through the pain, giving up hurts more.", char: "Vegeta", anime: "Dragon Ball Z" },
    { text: "A dropout will beat a genius through hard work.", char: "Rock Lee", anime: "Naruto" },
    { text: "It's not about whether I can. I have to do it.", char: "Megumi Fushiguro", anime: "Jujutsu Kaisen" },
    { text: "Hard work is worthless for those that don't believe in themselves.", char: "Naruto Uzumaki", anime: "Naruto" },
    { text: "Whatever you lose, you'll find it again. But what you throw away you'll never get back.", char: "Kenshin Himura", anime: "Rurouni Kenshin" },
    { text: "Fear is not evil. It tells you what your weakness is.", char: "Gildarts Clive", anime: "Fairy Tail" },
    { text: "I'll take a potato chip... and EAT IT!", char: "Light Yagami", anime: "Death Note" },
    { text: "In this world, those who break the rules are scum, that's true, but those who abandon their friends are worse than scum.", char: "Obito Uchiha", anime: "Naruto" },
];

const AnimeQuote = () => {
    const [quote] = useState(() => quotes[Math.floor(Math.random() * quotes.length)]);

    return (
        <div className="glass-card mb-6 border-l-2 border-white/40 bg-gradient-to-r from-white/5 to-transparent relative overflow-hidden">
            <div className="absolute right-0 top-0 h-full w-1/2 bg-gradient-to-l from-black/50 to-transparent pointer-events-none" />
            <div className="flex flex-col space-y-2 relative z-10">
                <p className="text-lg italic text-white/90 font-serif tracking-wide leading-relaxed">"{quote.text}"</p>
                <div className="flex items-center space-x-2 text-xs uppercase tracking-widest mt-2">
                    <span className="font-bold text-white">{quote.char}</span>
                    <span className="text-textMuted">•</span>
                    <span className="text-textMuted">{quote.anime}</span>
                </div>
            </div>
        </div>
    );
};

export default AnimeQuote;
